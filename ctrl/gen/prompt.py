# Copyright (2025) critic-rl Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
import re
from abc import ABC, abstractmethod

CRITIQUE_INSTRUCTION = "You are tasked with analyzing an answer to a problem and providing constructive feedback. Do NOT provide direct solutions."
CRITIQUE_WITH_HINT_INSTRUCTION = """You are tasked with analyzing an answer to a problem and providing constructive feedback. Do NOT provide direct solutions.
Please carefully reason about the hint to guide the user.
**Important: Do NOT mention 'the hint' in your feedback.**"""


def extract_critique_from_xml(xml_str: str) -> str:
    """Extract the critique from the xml string."""
    start_tag = "<format>"
    end_tag = "</format>"
    start_idx = xml_str.find(start_tag)
    end_idx = xml_str.find(end_tag)
    if start_idx == -1 or end_idx == -1:
        return xml_str
    return xml_str[start_idx + len(start_tag) : end_idx].strip()


def extract_error_from_traceback(traceback_str: str):
    traceback_str = traceback_str.strip()
    traceback_lines = traceback_str.splitlines()

    # error type is the last line of the traceback
    error_str = traceback_lines[-1].strip()

    # error location is the second line of the traceback
    location_str = traceback_lines[-2].strip()

    # format the error message
    error_message = f"""The code block
```python
{location_str}
```
raised {error_str}"""

    return error_message


def construct_hint_from_exec_for_io(eval_res: str) -> str:
    # first, check if the code passed all the tests
    accepted = eval_res["accepted"]
    if accepted:
        return "The provided answer is correct. A concise and positive response is recommended without unnecessary suggestions."

    # if the code failed all the tests, check if there are any errors
    test_res = eval_res["tests"]

    all_msg = []
    for test_dict in test_res:
        passed, exec_info, test_info = (
            test_dict["passed"],
            test_dict["exec_info"],
            test_dict["test_info"],
        )
        if passed:
            continue

        status = exec_info["run_result"]["status"]
        if status == "TimeLimitExceeded":
            msg = "The provided answer exceeds the time limit and needs to be approached in a more efficient manner."
            all_msg.append(msg)
            continue

        stderr = exec_info["run_result"]["stderr"]
        if "Traceback" in stderr:
            stderr = stderr[stderr.index("Traceback") :]
        else:
            stderr = ""
        # if there is an error, return the error message
        if len(stderr) > 0:
            msg = extract_error_from_traceback(stderr)
            return msg
        else:
            stdin = test_info["input"]["stdin"]
            expected_stdout = test_info["output"]["stdout"]
            actual_stdout = exec_info["run_result"]["stdout"]
            msg = f"""The provided answer is incorrect and does not pass the test case:
Input
{stdin.strip()}

Expected Output
{expected_stdout.strip()}

Actual Output
{actual_stdout.strip()}"""

        all_msg.append(msg)

    if len(all_msg) == len(test_res):
        return "The provided answer is completely incorrect and needs to be approached differently with a fresh start."

    all_msg = list(set(all_msg))
    all_msg = [e for e in all_msg if len(e) < 1000]

    if len(all_msg) == 0:
        return "The provided answer is completely incorrect and needs to be approached differently with a fresh start."

    return random.sample(all_msg, 1)[0]


def construct_hint_from_exec_for_fncall(eval_res: str) -> str:
    # first, check if the code passed all the tests
    status = eval_res["status"]
    if status == "Success":
        return "The provided answer is correct. A concise and positive response is recommended without unnecessary suggestions."

    run_result = eval_res["run_result"]
    if run_result["status"] == "TimeLimitExceeded":
        return "The provided answer exceeds the time limit and needs to be approached in a more efficient manner."

    stderr = run_result["stderr"]
    return f"The provided answer is incorrect and receives the following error:\n{stderr.strip()}"


class Prompter(ABC):
    @abstractmethod
    def get_gen_prompt(self, problem: str) -> str:
        """Generate a prompt for initial code generation."""
        pass

    @abstractmethod
    def get_critique_prompt(self, problem: str, solution: str) -> str:
        """Generate a prompt for critiquing a solution."""
        pass

    @abstractmethod
    def get_critique_with_exec_prompt(
        self, problem: str, solution: str, eval_res: dict
    ) -> str:
        """Generate a prompt for critiquing a solution with evaluation result."""
        pass

    @abstractmethod
    def get_revised_prompt_mt(self, critique: str) -> str:
        """Generate a prompt for revising a solution based on critique. This is for multi-turn interaction."""
        pass


class BasicPrompter(Prompter):
    def get_gen_prompt(self, problem: str) -> str:
        return problem

    def get_critique_prompt(self, problem: str, solution: str) -> str:
        return f"""{CRITIQUE_INSTRUCTION}

Problem description:
<problem>
{problem}
</problem>

Answer:
<answer>
{solution}
</answer>

Structure your response using the following format (without <format> tags):
<format>
Analysis:
{{Analysis}}

Improvement suggestions:
{{Suggestions}}

Overall judgment: {{Correct/Incorrect}}
</format>"""

    def get_revised_prompt_mt(self, critique: str) -> str:
        critique = extract_critique_from_xml(critique).strip()
        return f"""{critique}

Please finalize your answer accordingly using the same format."""

    def get_critique_with_exec_prompt(
        self, problem: str, solution: str, eval_res: dict
    ) -> str:
        raise NotImplementedError


class CodeContestsPrompter(Prompter):
    _gen_instruction = "Write python code to solve the following coding problem that obeys the constraints and passes the example test cases."
    _solution_format_info = "The output code needs to read from and write to standard IO. Please wrap your code answer using ```."

    def get_gen_prompt(self, problem: str) -> str:
        return f"""{self._gen_instruction}

Problem description:
<problem>
{problem}
</problem>

Format:
<format>
{self._solution_format_info}
</format>"""

    def get_critique_prompt(self, problem: str, solution: str) -> str:
        return f"""{CRITIQUE_INSTRUCTION}

Problem description:
<problem>
{problem}
</problem>

Answer:
<answer>
{solution}
</answer>

Structure your response using the following format (without <format> tags):
<format>
Analysis:
{{Analysis}}

Improvement suggestions:
{{Suggestions}}

Overall judgment: {{Correct/Incorrect}}
</format>"""

    def get_revised_prompt_mt(self, critique: str) -> str:
        critique = extract_critique_from_xml(critique).strip()
        return f"""{critique}

Please finalize your answer accordingly using the same format."""

    def get_critique_with_exec_prompt(
        self, problem: str, solution: str, eval_res: dict
    ) -> str:
        hint = construct_hint_from_exec_for_io(eval_res)

        return f"""{CRITIQUE_WITH_HINT_INSTRUCTION}

Problem description:
<problem>
{problem}
</problem>

Answer:
<answer>
{solution}
</answer>

Hint:
<hint>
{hint}
</hint>

Structure your response using the following format (without <format> tags):
<format>
Analysis:
{{Analysis}}

Improvement suggestions:
{{Suggestions}}

Overall judgment: {{Correct/Incorrect}}
</format>"""


class LiveCodeBenchPrompter(Prompter):
    def get_gen_prompt(self, problem: str) -> str:
        return problem

    def _extract_question(self, prompt: str):
        question = re.search(r"### Question:\n(.*)\n\n### Format:", prompt, re.DOTALL)
        start_code = re.search(
            r"### Format: .*\n```python\n(.*)\n```\n\n", prompt, re.DOTALL
        )
        assert question is not None and start_code is not None

        question = question.group(1)
        start_code = start_code.group(1)
        if start_code == "# YOUR CODE HERE":
            start_code = None
        return question, start_code

    def get_critique_prompt(self, problem: str, solution: str) -> str:
        question, _ = self._extract_question(problem)

        return f"""{CRITIQUE_INSTRUCTION}

Problem description:
<problem>
{question}
</problem>

Answer:
<answer>
{solution}
</answer>

Structure your response using the following format (without <format> tags):
<format>
Analysis:
{{Analysis}}

Improvement suggestions:
{{Suggestions}}

Overall judgment: {{Correct/Incorrect}}
</format>"""

    def get_revised_prompt_mt(self, critique: str) -> str:
        critique = extract_critique_from_xml(critique).strip()
        return f"""{critique}

Please finalize your answer accordingly using the same format."""

    def get_critique_with_exec_prompt(
        self, problem: str, solution: str, eval_res: dict
    ) -> str:
        raise NotImplementedError()


class EvalPlusPrompter(Prompter):
    _gen_instruction = "Write python code to solve the following coding problem."
    _solution_format_info = "The output code should not include test cases. Please wrap your code answer using ```."

    def get_gen_prompt(self, problem: str) -> str:
        return f"""{self._gen_instruction}

Problem description:
<problem>
{problem}
</problem>

Format:
<format>
{self._solution_format_info}
</format>"""

    def get_critique_prompt(self, problem: str, solution: str) -> str:
        return f"""{CRITIQUE_INSTRUCTION}

Problem description:
<problem>
{problem}
</problem>

Answer:
<answer>
{solution}
</answer>

Structure your response using the following format (without <format> tags):
<format>
Analysis:
{{Analysis}}

Improvement suggestions:
{{Suggestions}}

Overall judgment: {{Correct/Incorrect}}
</format>"""

    def get_revised_prompt_mt(self, critique: str) -> str:
        return f"""{extract_critique_from_xml(critique).strip()}

Please finalize your answer accordingly using the same format."""

    def get_critique_with_exec_prompt(
        self, problem: str, solution: str, eval_res: dict
    ) -> str:
        hint = construct_hint_from_exec_for_fncall(eval_res)

        return f"""{CRITIQUE_WITH_HINT_INSTRUCTION}

Problem description:
<problem>
{problem}
</problem>

Answer:
<answer>
{solution}
</answer>

Hint:
<hint>
{hint}
</hint>

Structure your response using the following format (without <format> tags):
<format>
Analysis:
{{Analysis}}

Improvement suggestions:
{{Suggestions}}

Overall judgment: {{Correct/Incorrect}}
</format>"""


def get_prompter(dataset_type: str) -> Prompter:
    if dataset_type == "code_contests":
        return CodeContestsPrompter()
    elif dataset_type == "livecodebench":
        return LiveCodeBenchPrompter()
    elif dataset_type == "mbppplus":
        return EvalPlusPrompter()
    else:
        return BasicPrompter()
