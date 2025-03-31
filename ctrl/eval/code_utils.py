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
import re


def sanitize(text):
    # Pattern to match code blocks starting with ```, with optional language identifier
    # and capturing everything after until the end or until another ```
    pattern = r"```(?:python)?\s*([\s\S]*?)(?:\s*```|$)"
    # Find all matches in the text
    matches = re.findall(pattern, text, re.IGNORECASE)

    # Return the first one
    return f"```python\n{matches[0]}\n```" if matches and len(matches[0]) > 0 else text


def desanitize(text):
    # Remove the starting and ending ```
    pattern = r"```(?:python)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        return match.group(1).strip()
    return text
