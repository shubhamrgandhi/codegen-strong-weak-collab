AGENTLESS_PROMPT = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Below are some code segments, each from a relevant file. One or more of these files may contain bugs:
--- BEGIN FILE ---
{retrieval}
--- END FILE ---

Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.
"""

AGENTLESS_PROMPT_WITH_TRAJECTORY = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

An expert has already worked on this issue. Here are all of its analyses, attempts and solutions:
--- BEGIN EXPERT SOLUTIONS AND ATTEMPTS ---
{trajectory}
--- END EXPERT SOLUTIONS AND ATTEMPTS ---

Below are some code segments, each from a relevant file. One or more of these files may contain bugs:
--- BEGIN FILE ---
{retrieval}
--- END FILE ---

Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.
"""

AGENTLESS_PROMPT_WITH_PLAN = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Here is a plan that you can follow to solve this issue:
--- BEGIN PLAN ---
{plan}
--- END PLAN ---

Below are some code segments, each from a relevant file. One or more of these files may contain bugs:
--- BEGIN FILE ---
{retrieval}
--- END FILE ---

Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.
"""


AGENTLESS_PROMPT_WITH_INFO = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Here is some high-level information about the repository that you might find useful to solve this issue:
--- BEGIN REPOSITORY INFO ---
{info}
--- END REPOSITORY INFO ---

Below are some code segments, each from a relevant file. One or more of these files may contain bugs:
--- BEGIN FILE ---
{retrieval}
--- END FILE ---

Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.
"""


AGENTLESS_PROMPT_WITH_REPO_FAQ = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Here are some FAQs about the repository that you might find useful to solve this issue:
--- BEGIN REPOSITORY FAQs ---
{repo_faq}
--- END REPOSITORY FAQs ---

Below are some code segments, each from a relevant file. One or more of these files may contain bugs:
--- BEGIN FILE ---
{retrieval}
--- END FILE ---

Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.
"""


AGENTLESS_PROMPT_WITH_INSTANCE_FAQ = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Here are some FAQs to help you solve this issue:
--- BEGIN FAQs ---
{instance_faq}
--- END FAQs ---

Below are some code segments, each from a relevant file. One or more of these files may contain bugs:
--- BEGIN FILE ---
{retrieval}
--- END FILE ---

Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.
"""


AGENTLESS_PROMPT_WITH_FEW_SHOT = """
Here are some {similar}example issues from the same repository along with the target file that were changed and final patch generated by an expert{successful}:
--- BEGIN EXAMPLES ---
{few_shot_examples}
--- END EXAMPLES ---

We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Below are some code segments, each from a relevant file. One or more of these files may contain bugs:
--- BEGIN FILE ---
{retrieval}
--- END FILE ---

Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.
"""


REDUCTION_PROMPT = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Below are some code segments, each from a relevant file. One or more of these files may contain bugs:
--- BEGIN FILE ---
{retrieval}
--- END FILE ---

Your task is to identify the most relevant code sections that contain the bug or need to be modified to fix the issue.
Please output only the file paths and relevant code sections in this format:

### <file_path>
```python
# relevant code section 1
```

### <file_path>
```python
# relevant code section 2
```

Only include the minimum necessary code with sufficient context to understand and fix the issue. Ensure that the code is complete and valid Python code.
Don't include any explanations or reasoning - just the file paths and code sections.
"""

REPO_INSIGHT_PROMPT = """I need you to provide high-level insights about the following repository: {repo_name}

Based on the repository structure and README below, generate a comprehensive overview of this repository that could help guide a language model in solving technical issues.

Repository Structure:
{repo_structure}

README Content:
{readme_content}

Please provide the following insights. For each point, provide concrete details and specific examples from the codebase - high-level doesn't mean vague, it means providing a clear architectural overview with specific names, patterns, and implementations:

1. Core Purpose and Functionality: 
    - What specific problem does this repository solve?
    - What are its primary features and capabilities?

2. Main Architectural Patterns:
    - Identify concrete architectural patterns used in this codebase
    - EXAMPLE: Plugin based architecture, layered architecture, etc

3. Module Organization:
    - Name the specific key modules and their exact responsibilities
    - EXAMPLE: I/O module, error-handling module, etc

4. Key Abstractions and Concepts:
    - List the actual fundamental abstractions used in the codebase
    - EXAMPLE: Quantity class for numerical values, Logger class for logging, etc

5. Design Patterns:
    - Identify specific recurring code patterns with examples
    - EXAMPLE: Factory methods, Decorators, etc

6. Error Handling Approaches:
    - Describe precise error handling mechanisms used in the codebase
    - EXAMPLE: Custom exception hierarchies, warnings, etc

Focus on providing actionable architectural insights that would be valuable for understanding the repository's design philosophy and core abstractions. Your response should contain specific implementation details that would help someone understand how to navigate, extend, and debug the codebase to solve issues.
"""


REPO_FAQ_PROMPT = """I need you to generate a comprehensive FAQ about the repository: {repo_name}

Based on the repository structure and README below, create a detailed set of technical FAQs that would help a developer solve issues in this codebase. These FAQs should serve as guidance for someone who is trying to resolve bugs or implement new features.

Repository Structure:
{repo_structure}

README Content:
{readme_content}

Please generate 15-20 frequently asked questions with detailed answers about:

1. Code Organization and Architecture:
   - How is the codebase structured?
   - What are the key modules and their responsibilities?
   - How do the different components interact?

2. Common Patterns and Conventions:
   - What design patterns are commonly used?
   - What are the naming conventions and code style expectations?
   - Are there specific patterns for implementing new features?

3. Typical Debugging Approaches:
   - What are common error patterns and their solutions?
   - How to debug specific types of issues in this codebase?
   - What are common pitfalls when modifying this code?

4. Implementation Details:
   - How are core abstractions implemented?
   - What are the key algorithms or data structures used?
   - How does the error handling system work?

5. Testing Considerations:
   - How is testing typically done in this codebase?
   - What should be considered when writing tests?
   - Are there common test fixtures or utilities?

For each question, provide detailed, specific answers with concrete examples from the codebase when possible. Focus on information that would be most valuable to someone trying to fix bugs or implement new features. The FAQs should reflect the actual patterns and practices used in this specific repository, not generic software development advice.
"""