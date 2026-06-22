---
name: comment-not-delete
description: When asked to "comment out" code, keep the original lines as comments — do not delete them
metadata:
  type: feedback
---

When asked to "comment out" code, preserve the original lines as comments in place. Do not delete them or rewrite the function without them.

**Why:** The user wants to be able to easily restore the code later by uncommenting.

**How to apply:** Use `#` to prefix each line of the code being disabled. Keep the commented-out block in its original location.
