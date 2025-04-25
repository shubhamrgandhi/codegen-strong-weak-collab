import sys

fname = "data/repair_cp.log"
content = open(fname).read()
lines = content.split("\n")

issue2content = {}
cur_lines = []
issue_name = ""
prompt_mode = False
for lid,line in enumerate(lines):
    if "INFO - ================ repairing" in line:
        assert len(cur_lines) == 0
        issue_name = line.split("repairing ")[1].split(" ================")[0]
        cur_lines.append(line)
        prompt_mode = True
    elif "skipped since no files were localized" in line or (line[:5] != "2024-" and "Wrap the *SEARCH/REPLACE* edit in blocks ```python...```." in line):
        assert prompt_mode
        cur_lines.append(line)
        issue2content[issue_name] = cur_lines
        cur_lines = []
        issue_name = ""
        prompt_mode = False
    elif prompt_mode:
        cur_lines.append(line)


assert len(issue2content) == 300
issue2prompt = {}
for issue in issue2content:
    content = "\n".join(issue2content[issue])
    if "skipped since no files were localized" in content:
        issue2prompt[issue] = "skipped since no files were localized"
    else:
        assert "prompting with message:" in content
        issue2prompt[issue] = content.split("prompting with message:")[1].strip()

import pickle
pickle.dump(issue2prompt, open("data/issue2repograph_prompt.pkl",'wb'))

