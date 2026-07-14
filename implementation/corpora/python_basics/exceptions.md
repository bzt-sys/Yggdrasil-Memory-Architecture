# Python exception basics

NameError usually means code referenced a name that is not defined in the current namespace. Repairs include defining the helper, importing the symbol explicitly, or replacing the dependency with local logic.

AssertionError in a test harness means the code executed but did not satisfy an expected condition. It does not by itself reveal which semantic assumption was false. Useful next evidence includes a failing input, expected output, and observed output.

SyntaxError means the submitted text was not valid Python. Common causes include truncated code, prose accidentally included in executable text, unclosed blocks, or invalid indentation.
