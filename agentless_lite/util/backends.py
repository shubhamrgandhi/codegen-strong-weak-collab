import base64
import json
import os
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from io import BytesIO

import anthropic
import openai
import requests
from PIL import Image

from agentless_lite.util.logging import setup_logging
from agentless_lite.util.repair import create_diff_from_response, fake_git_repo

STR_REPLACE_EDITOR_DESCRIPTION = """Custom editing tool for editing files
* State is persistent across command calls and discussions with the user

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
"""

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "str_replace_editor",
            "description": STR_REPLACE_EDITOR_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "description": "Full path to file, e.g. `folder/file.py`.",
                        "type": "string",
                    },
                    "old_str": {
                        "description": "Required parameter containing the string in `path` to replace.",
                        "type": "string",
                    },
                    "new_str": {
                        "description": "Optional parameter containing the new string (if not given, no string will be added).",
                        "type": "string",
                    },
                },
                "required": ["path", "old_str"],
                "additionalProperties": False,
            },
        },
    }
]

DEEPSEEK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "str_replace_editor",
            "description": STR_REPLACE_EDITOR_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "description": "Full path to file, e.g. `folder/file.py`.",
                        "type": "string",
                    },
                    "old_str": {
                        "description": "Required parameter containing the string in `path` to replace.",
                        "type": "string",
                    },
                    "new_str": {
                        "description": "Optional parameter containing the new string (if not given, no string will be added).",
                        "type": "string",
                    },
                },
                "required": ["path", "old_str"],
            },
        },
    }
]

ANTHROPIC_TOOLS = [
    {
        "name": "str_replace_editor",
        "description": STR_REPLACE_EDITOR_DESCRIPTION,
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "description": "Full path to file, e.g. `folder/file.py`.",
                    "type": "string",
                },
                "old_str": {
                    "description": "Required parameter containing the string in `path` to replace.",
                    "type": "string",
                },
                "new_str": {
                    "description": "Optional parameter containing the new string (if not given, no string will be added).",
                    "type": "string",
                },
            },
            "required": ["path", "old_str"],
            "cache_control": {"type": "ephemeral"},
        },
    }
]


class CodeGenerator(ABC):
    @abstractmethod
    def generate(
        self, instance, prompt, args, file_lock, output_file, image_assets=None
    ):
        pass

    @abstractmethod
    def initialize_output_files(self, args):
        pass


def request_chatgpt_engine(config, logger, base_url=None, max_retries=40, timeout=100):
    ret = None
    retries = 0

    client = openai.OpenAI(
        base_url=base_url,
    )

    while ret is None and retries < max_retries:
        try:
            logger.info(f"Creating API request: \n\n{config}")
            ret = client.chat.completions.create(**config)

            if ret is None or not hasattr(ret, "choices"):
                logger.error(f"Invalid response received: {ret}")
                raise Exception("Invalid API response")

        except openai.OpenAIError as e:
            if isinstance(e, openai.BadRequestError):
                logger.error("Request invalid")
                logger.error(str(e))
                raise Exception("Invalid API Request")
            elif isinstance(e, openai.RateLimitError):
                logger.info("Rate limit exceeded. Waiting...")
                logger.error(str(e))
                time.sleep(5)
            elif isinstance(e, openai.APIConnectionError):
                logger.info("API connection error. Waiting...")
                logger.error(str(e))
                time.sleep(5)
            elif isinstance(
                e, openai.APITimeoutError
            ):  # Add specific handling for timeout
                logger.info(f"Request timed out after {timeout} seconds. Retrying...")
                logger.error(str(e))
                time.sleep(1)
            else:
                logger.info("Unknown error. Waiting...")
                logger.error(str(e))
                time.sleep(1)

        retries += 1
        if retries >= max_retries:
            logger.error(f"Max retries ({max_retries}) exceeded")
            ret = None

    logger.info(f"API response {ret}")
    return ret


class BaseGenerator(CodeGenerator):
    def get_temperature_for_generation(self, gen_index, max_temp, total_gens):
        if not hasattr(self, "temperature_list"):
            temp_list = [0.0]  # First generation at 0

            for temp in [0.2, 0.4, 0.6, 0.8]:
                temp_list.extend([temp] * 4)

            for temp in [1.0, 1.2]:
                temp_list.extend([temp] * 8)

            current_temp = 1.4
            while len(temp_list) < total_gens:
                temp_to_add = min(current_temp, max_temp)
                temp_list.extend([temp_to_add] * 8)
                current_temp += 0.2

            temp_list = temp_list[:total_gens]
            self.temperature_list = temp_list

        return self.temperature_list[gen_index]

    def initialize_output_files(self, args):
        if not os.path.exists(args.output_file):
            with open(args.output_file, "w", encoding="utf-8") as outfile:
                pass

    def get_existing_entry(self, instance, file_lock, output_file):
        with file_lock:
            with open(output_file, "r", encoding="utf-8") as infile:
                for line in infile:
                    try:
                        entry = json.loads(line.strip())
                        if entry["instance_id"] == instance["instance_id"]:
                            return entry
                    except json.JSONDecodeError:
                        continue
        return None

    def update_output_file(self, output_entry, instance, file_lock, output_file):
        with file_lock:
            with open(output_file, "r", encoding="utf-8") as infile:
                lines = infile.readlines()

            entry_updated = False
            for j, line in enumerate(lines):
                try:
                    entry = json.loads(line.strip())
                    if entry["instance_id"] == instance["instance_id"]:
                        lines[j] = json.dumps(output_entry) + "\n"
                        entry_updated = True
                        break
                except json.JSONDecodeError:
                    continue

            if not entry_updated:
                lines.append(json.dumps(output_entry) + "\n")

            with open(output_file, "w", encoding="utf-8") as outfile:
                outfile.writelines(lines)

    def generate_with_retries(
        self, instance, prompt, args, file_lock, output_file, image_assets=None
    ):
        # Create a thread-local copy of args
        local_args = deepcopy(args)

        # Get logs directory relative to output file
        logs_dir = os.path.join(os.path.dirname(output_file), "logs")
        logger = setup_logging(instance["instance_id"], logs_dir)

        temperature = local_args.temp
        local_args.max_retries = 1

        for attempt in range(args.max_retries):
            try:
                local_args.temp = temperature
                response, output_entry = self.generate(
                    instance,
                    prompt,
                    local_args,
                    file_lock,
                    output_file,
                    defer_writing=True,
                    image_assets=image_assets,
                )

                # Handle both string responses and tool use responses
                git_diff = None
                if isinstance(response, str):
                    git_diff = create_diff_from_response(
                        response, instance["file_contents"], instance["found_files"]
                    )
                elif isinstance(response, list):  # Tool use response
                    # Apply edits directly from tool use commands
                    new_contents = instance["file_contents"].copy()
                    for path, old_str, new_str in response:
                        file_idx = instance["found_files"].index(path)
                        if file_idx >= 0:
                            new_contents[file_idx] = new_contents[file_idx].replace(
                                old_str, new_str
                            )
                    git_diff = fake_git_repo(
                        "playground",
                        instance["found_files"],
                        instance["file_contents"],
                        new_contents,
                    )

                if git_diff:
                    # Write both the generation output and the diff
                    with file_lock:
                        with open(output_file, "a", encoding="utf-8") as f:
                            output = {
                                "instance_id": instance["instance_id"],
                                "model_name_or_path": "agentless_lite_greedy",
                                "model_patch": git_diff,
                                "temperature": temperature,
                                "attempt": attempt + 1,
                            }
                            f.write(json.dumps(output) + "\n")

                    return git_diff

                temperature = min(1.0, temperature + 0.1)
            except Exception as e:
                logger.error(f"Error in generation attempt {attempt + 1}: {e}")
                temperature = min(1.0, temperature + 0.1)

        return False


class OpenAIGenerator(BaseGenerator):
    def generate(
        self,
        instance,
        prompt,
        args,
        file_lock,
        output_file,
        defer_writing=False,
        image_assets=None,
    ):
        logs_dir = os.path.join(os.path.dirname(output_file), "logs")
        logger = setup_logging(instance["instance_id"], logs_dir)
        logger.info("Initializing OpenAI client")

        existing_entry = self.get_existing_entry(instance, file_lock, output_file)
        all_responses = [] if not existing_entry else existing_entry["responses"]

        all_usage = (
            existing_entry["usage"]
            if existing_entry
            else (
                {"completion_tokens": 0, "prompt_tokens": 0, "logprobs": [], "temp": []}
                if args.logprobs
                else {"completion_tokens": 0, "prompt_tokens": 0, "temp": []}
            )
        )

        start_index = len(all_responses)

        if args.warming:
            # Group generations by temperature
            temp_groups = {}
            for i in range(start_index, args.max_retries):
                temperature = self.get_temperature_for_generation(
                    i, args.temp, args.max_retries
                )
                if temperature not in temp_groups:
                    temp_groups[temperature] = []
                temp_groups[temperature].append(i)

            # Generate samples for each temperature group
            for temperature, indices in temp_groups.items():
                num_samples = len(indices)
                logger.info(
                    f"Making API call for {num_samples} samples with temperature {temperature}"
                )

                if "o3-mini" in args.model:
                    config = {
                        "model": args.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "n": args.max_retries - start_index,
                        "max_completion_tokens": args.max_completion_tokens,
                        "response_format": {"type": "text"},
                        "reasoning_effort": "high",
                        "store": True,
                    }
                else:
                    config = {
                        "model": args.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "n": args.max_retries - start_index,
                        "temperature": temperature,
                        "max_tokens": args.max_completion_tokens,
                        "logprobs": args.logprobs,
                        "store": True,
                    }

                if image_assets:
                    content = [{"type": "text", "text": prompt}]
                    for idx, image in enumerate(image_assets):
                        image_base64 = get_base64_from_url(image)
                        image_data = {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_base64,
                            },
                        }
                        if idx < 1:  # Only add cache_control for first two images
                            image_data["source"]["cache_control"] = {
                                "type": "ephemeral"
                            }
                        content.append(image_data)
                    config["messages"] = [{"role": "user", "content": content}]

                completion = request_chatgpt_engine(config, logger)

                if completion is None:
                    raise Exception("Failed to get response from API")

                if args.logprobs:
                    logprobs_data = [
                        {
                            "token": lp.token,
                            "logprob": lp.logprob,
                            "bytes": lp.bytes,
                            "top_logprobs": lp.top_logprobs,
                        }
                        for lp in completion.choices[0].logprobs.content
                    ]
                    all_usage["logprobs"].extend([logprobs_data] * num_samples)

                all_responses.extend(
                    [choice.message.content for choice in completion.choices]
                )

                # Update usage information
                all_usage["completion_tokens"] += completion.usage.completion_tokens
                all_usage["prompt_tokens"] += completion.usage.prompt_tokens
                all_usage["temp"].extend([temperature] * num_samples)
                if args.logprobs and hasattr(completion, "logprobs"):
                    all_usage["logprobs"].extend([completion.logprobs] * num_samples)

                output_entry = {
                    "instance_id": instance["instance_id"],
                    "found_files": instance["found_files"],
                    "file_contents": instance["file_contents"],
                    "responses": all_responses,
                    "usage": all_usage,
                }

                if not defer_writing:
                    self.update_output_file(
                        output_entry, instance, file_lock, output_file
                    )
        else:
            # Generate all samples in one batch
            temperature = args.temp
            logger.info(f"Making batch API call with temperature {temperature}")

            if "o3-mini" in args.model:
                config = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "n": args.max_retries - start_index,
                    "max_completion_tokens": args.max_completion_tokens,
                    "response_format": {"type": "text"},
                    "reasoning_effort": "high",
                    "store": True,
                }
            else:
                config = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "n": args.max_retries - start_index,
                    "temperature": temperature,
                    "max_tokens": args.max_completion_tokens,
                    "logprobs": args.logprobs,
                    "store": True,
                }

            if image_assets:
                for idx, image in enumerate(image_assets):
                    image_base64 = get_base64_from_url(image)
                    image_data = {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64,
                        },
                    }
                    if idx < 2:  # Only add cache_control for first two images
                        image_data["source"]["cache_control"] = {"type": "ephemeral"}
                    content = [{"type": "text", "text": prompt}]
                    content.append(image_data)
                    config["messages"] = [{"role": "user", "content": content}]

            completion = request_chatgpt_engine(config, logger)

            if completion is None:
                raise Exception("Failed to get response from API")

            if args.logprobs:
                logprobs_data = [
                    {
                        "token": lp.token,
                        "logprob": lp.logprob,
                        "bytes": lp.bytes,
                        "top_logprobs": lp.top_logprobs,
                    }
                    for lp in completion.choices[0].logprobs.content
                ]
                all_usage["logprobs"].extend(
                    [logprobs_data] * (args.max_retries - start_index)
                )

            all_responses.extend(
                [choice.message.content for choice in completion.choices]
            )

            all_usage["completion_tokens"] += completion.usage.completion_tokens
            all_usage["prompt_tokens"] += completion.usage.prompt_tokens
            all_usage["temp"].extend([temperature] * (args.max_retries - start_index))
            if args.logprobs and hasattr(completion, "logprobs"):
                all_usage["logprobs"].extend(completion.logprobs)

            output_entry = {
                "instance_id": instance["instance_id"],
                "found_files": instance["found_files"],
                "file_contents": instance["file_contents"],
                "responses": all_responses,
                "usage": all_usage,
            }

            if not defer_writing:
                self.update_output_file(output_entry, instance, file_lock, output_file)

        logger.info("Output written successfully")
        return all_responses[-1], output_entry
    
    def generate_plan(
        self,
        instance,
        prompt,
        args,
        file_lock,
        output_file,
        defer_writing=False,
        image_assets=None,
    ):
        logs_dir = os.path.join(os.path.dirname(output_file), "logs")
        logger = setup_logging(instance["instance_id"], logs_dir)
        logger.info("Initializing OpenAI client")

        existing_entry = self.get_existing_entry(instance, file_lock, output_file)
        all_responses = [] if not existing_entry else existing_entry["responses"]

        all_usage = (
            existing_entry["usage"]
            if existing_entry
            else (
                {"completion_tokens": 0, "prompt_tokens": 0, "logprobs": [], "temp": []}
                if args.logprobs
                else {"completion_tokens": 0, "prompt_tokens": 0, "temp": []}
            )
        )

        start_index = len(all_responses)

        # Generate all samples in one batch
        temperature = args.temp
        logger.info(f"Making batch API call with temperature {temperature}")

        if "o3-mini" in args.model:
            config = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "n": args.max_retries - start_index,
                "max_completion_tokens": args.max_completion_tokens,
                "response_format": {"type": "text"},
                "reasoning_effort": "high",
                "store": True,
            }
        else:
            config = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "n": args.max_retries - start_index,
                "temperature": temperature,
                "max_tokens": args.max_completion_tokens,
                "logprobs": args.logprobs,
                "store": True,
            }

        completion = request_chatgpt_engine(config, logger)

        if completion is None:
            raise Exception("Failed to get response from API")

        if args.logprobs:
            logprobs_data = [
                {
                    "token": lp.token,
                    "logprob": lp.logprob,
                    "bytes": lp.bytes,
                    "top_logprobs": lp.top_logprobs,
                }
                for lp in completion.choices[0].logprobs.content
            ]
            all_usage["logprobs"].extend(
                [logprobs_data] * (args.max_retries - start_index)
            )

        all_responses.extend(
            [choice.message.content for choice in completion.choices]
        )

        all_usage["completion_tokens"] += completion.usage.completion_tokens
        all_usage["prompt_tokens"] += completion.usage.prompt_tokens
        all_usage["temp"].extend([temperature] * (args.max_retries - start_index))
        if args.logprobs and hasattr(completion, "logprobs"):
            all_usage["logprobs"].extend(completion.logprobs)

        output_entry = {
            "instance_id": instance["instance_id"],
            "plan": all_responses[-1],
            "usage": all_usage,
        }

        if not defer_writing:
            self.update_output_file(output_entry, instance, file_lock, output_file)

        logger.info("Output written successfully")
        return all_responses[-1], output_entry


class DeepSeekGenerator(BaseGenerator):
    def generate(
        self,
        instance,
        prompt,
        args,
        file_lock,
        output_file,
        defer_writing=False,
        image_assets=None,
    ):
        logs_dir = os.path.join(os.path.dirname(output_file), "logs")
        logger = setup_logging(instance["instance_id"], logs_dir)
        logger.info("Preparing DeepSeek configuration")

        existing_entry = self.get_existing_entry(instance, file_lock, output_file)
        all_responses = [] if not existing_entry else existing_entry["responses"]

        all_usage = (
            existing_entry["usage"]
            if existing_entry
            else (
                {"completion_tokens": 0, "prompt_tokens": 0, "logprobs": [], "temp": []}
                if args.logprobs
                else {"completion_tokens": 0, "prompt_tokens": 0, "temp": []}
            )
        )

        # Make multiple API calls based on max_retries
        start_index = len(all_responses)
        for i in range(start_index, args.max_retries):
            temperature = (
                self.get_temperature_for_generation(i, args.temp, args.max_retries)
                if args.warming
                else args.temp
            )
            logger.info(
                f"Making API call {i+1}/{args.max_retries} with temperature {temperature}"
            )

            if args.model == "deepseek-reasoner":
                config = {
                    "model": args.model,
                    "max_tokens": args.max_completion_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                }
            else:
                config = {
                    "model": args.model,
                    "max_tokens": args.max_completion_tokens,
                    "temperature": temperature,
                    "n": 1,
                    "logprobs": args.logprobs,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                }

            if image_assets:
                logger.info(type(image_assets))
                logger.info(image_assets)
                content = [{"type": "text", "text": prompt}]
                for image in image_assets:
                    content.append({"type": "image_url", "image_url": {"url": image}})
                config["messages"] = [{"role": "user", "content": content}]

            if args.tool_use:
                config.update(
                    {
                        "tools": DEEPSEEK_TOOLS,
                        "tool_choice": "required",
                    }
                )

            gener = request_chatgpt_engine(
                config, logger=logger, base_url="https://api.deepseek.com"
            )

            if gener:
                response = (
                    gener.choices[0].message.tool_calls[0]
                    if args.tool_use
                    else gener.choices[0].message.content
                )
                all_responses.append(response)
                if args.logprobs:
                    logprobs_data = [
                        {
                            "token": lp.token,
                            "logprob": lp.logprob,
                            "bytes": lp.bytes,
                            "top_logprobs": lp.top_logprobs,
                        }
                        for lp in gener.choices[0].logprobs.content
                    ]
                    all_usage["logprobs"].append(logprobs_data)
                if args.warming:
                    all_usage["temp"].append(temperature)

            else:
                all_responses.append("")

            output_entry = {
                "instance_id": instance["instance_id"],
                "found_files": instance["found_files"],
                "file_contents": instance["file_contents"],
                "responses": all_responses,
                "usage": all_usage,
            }

            if not defer_writing:
                self.update_output_file(output_entry, instance, file_lock, output_file)

        logger.info("Output written successfully")
        return all_responses[-1], output_entry


class OpenRouterGenerator(BaseGenerator):
    def generate(
        self,
        instance,
        prompt,
        args,
        file_lock,
        output_file,
        defer_writing=False,
        image_assets=None,
    ):
        logs_dir = os.path.join(os.path.dirname(output_file), "logs")
        logger = setup_logging(instance["instance_id"], logs_dir)
        logger.info("Preparing OpenRouter configuration")

        existing_entry = self.get_existing_entry(instance, file_lock, output_file)
        all_responses = [] if not existing_entry else existing_entry["responses"]

        all_usage = (
            existing_entry["usage"]
            if existing_entry
            else (
                {"completion_tokens": 0, "prompt_tokens": 0, "logprobs": [], "temp": []}
                if args.logprobs
                else {"completion_tokens": 0, "prompt_tokens": 0, "temp": []}
            )
        )

        # Make multiple API calls based on max_retries
        start_index = len(all_responses)
        for i in range(start_index, args.max_retries):
            temperature = (
                self.get_temperature_for_generation(i, args.temp, args.max_retries)
                if args.warming
                else args.temp
            )
            logger.info(
                f"Making API call {i+1}/{args.max_retries} with temperature {temperature}"
            )

            config = {
                "model": args.model,
                "temperature": temperature,
                "n": 1,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
            }

            if image_assets:
                for image in image_assets:
                    content = [{"type": "text", "text": prompt}]
                    for image in image_assets:
                        content.append(
                            {"type": "image_url", "image_url": {"url": image}}
                        )
                    config["messages"] = [{"role": "user", "content": content}]

            if args.tool_use:
                config.update(
                    {
                        "tools": DEEPSEEK_TOOLS,
                        "tool_choice": "required",
                    }
                )

            gener = request_chatgpt_engine(
                config, logger=logger, base_url="https://openrouter.ai/api/v1"
            )

            if gener:
                response = (
                    gener.choices[0].message.tool_calls[0]
                    if args.tool_use
                    else gener.choices[0].message.content
                )
                all_responses.append(response)
            else:
                all_responses.append("")
                if args.logprobs:
                    logprobs_data = [
                        {
                            "token": lp.token,
                            "logprob": lp.logprob,
                            "bytes": lp.bytes,
                            "top_logprobs": lp.top_logprobs,
                        }
                        for lp in gener.choices[0].logprobs.content
                    ]
                    all_usage["logprobs"].append(logprobs_data)
                if args.warming:
                    all_usage["temp"].append(temperature)

            output_entry = {
                "instance_id": instance["instance_id"],
                "found_files": instance["found_files"],
                "file_contents": instance["file_contents"],
                "responses": all_responses,
                "usage": all_usage,
            }

            if not defer_writing:
                self.update_output_file(output_entry, instance, file_lock, output_file)

        logger.info("Output written successfully")
        return all_responses[-1], output_entry


def get_base64_from_url(image_url):
    """Convert an image URL to base64 encoding"""
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    buffered = BytesIO()
    img.save(buffered, format="PNG")

    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def request_anthropic_engine(config, logger, max_retries=40):
    ret = None
    retries = 0
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    while ret is None and retries < max_retries:
        try:
            logger.info("Creating Anthropic API request")
            ret = client.messages.create(**config)

            if ret is None:
                logger.error(f"Invalid response received: {ret}")
                raise Exception("Invalid API response")

        except anthropic.APIError as e:
            if isinstance(e, anthropic.BadRequestError):
                logger.error("Request invalid")
                logger.error(str(e))
                raise Exception("Invalid API Request")
            elif isinstance(e, anthropic.RateLimitError):
                logger.info("Rate limit exceeded. Waiting...")
                logger.error(str(e))
                time.sleep(5)
            elif isinstance(e, anthropic.APIConnectionError):
                logger.info("API connection error. Waiting...")
                logger.error(str(e))
                time.sleep(5)
            else:
                logger.info("Unknown error. Waiting...")
                logger.error(str(e))
                time.sleep(1)

        retries += 1
        if retries >= max_retries:
            logger.error(f"Max retries ({max_retries}) exceeded")
            ret = None

    logger.info(f"API response {ret}")
    return ret


class AnthropicGenerator(BaseGenerator):
    def generate(
        self,
        instance,
        prompt,
        args,
        file_lock,
        output_file,
        defer_writing=False,
        image_assets=None,
    ):
        logs_dir = os.path.join(os.path.dirname(output_file), "logs")
        logger = setup_logging(instance["instance_id"], logs_dir)
        logger.info("Initializing Anthropic client")

        existing_entry = self.get_existing_entry(instance, file_lock, output_file)
        all_responses = [] if not existing_entry else existing_entry["responses"]

        all_usage = (
            existing_entry["usage"]
            if existing_entry
            else {"completion_tokens": 0, "prompt_tokens": 0, "temp": []}
        )

        start_index = len(all_responses)
        for i in range(start_index, args.max_retries):
            temperature = (
                self.get_temperature_for_generation(i, args.temp, args.max_retries)
                if args.warming
                else args.temp
            )
            logger.info(
                f"Making API call {i+1}/{args.max_retries} with temperature {temperature}"
            )

            config = {
                "model": args.model,
                "max_tokens": args.max_completion_tokens,
                "temperature": temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                ],
            }

            if args.model == "claude-3-7-sonnet-20250219":
                config["temperature"] = 1
                config["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": args.max_completion_tokens - 4096,
                }

            if args.tool_use:
                config["tools"] = ANTHROPIC_TOOLS

            if image_assets:
                content = config["messages"][0]["content"]
                for image in image_assets:
                    image_base64 = get_base64_from_url(image)
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_base64,
                                # "cache_control": {"type": "ephemeral"}
                            },
                        }
                    )

            completion = request_anthropic_engine(config, logger)

            if completion:
                if args.tool_use:
                    logger.info("Processing tool use response")
                    logger.info(f"Raw completion: {completion}")
                    logger.info(f"Content blocks: {completion.content}")

                    # Extract tool use commands
                    edits = []
                    for block in completion.content:
                        logger.info(f"Processing content block: {block}")
                        logger.info(f"Block type: {block.type}")

                        if block.type == "tool_use":
                            params = block.input
                            logger.info(f"Tool call parameters: {params}")
                            edits.append(
                                (
                                    params["path"],
                                    params["old_str"],
                                    params.get("new_str", ""),
                                )
                            )
                            logger.info(f"Added edit: {edits[-1]}")

                    logger.info(f"Final collected edits: {edits}")
                    response = edits
                else:
                    response = completion.content[0].text

                all_responses.append(response)
                if args.warming:
                    all_usage["temp"].append(temperature)
            else:
                all_responses.append("" if not args.tool_use else [])

            output_entry = {
                "instance_id": instance["instance_id"],
                "found_files": instance["found_files"],
                "file_contents": instance["file_contents"],
                "responses": all_responses,
                "usage": all_usage,
            }

            if not defer_writing:
                self.update_output_file(output_entry, instance, file_lock, output_file)

        logger.info("Output written successfully")
        return all_responses[-1], output_entry


def get_generator(backend_type):
    generators = {
        "openai": OpenAIGenerator(),
        "deepseek": DeepSeekGenerator(),
        "open_router": OpenRouterGenerator(),
        "anthropic": AnthropicGenerator(),
    }
    return generators.get(backend_type)
