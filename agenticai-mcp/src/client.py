"""
Example MCP-style client calling the tool server.
"""

import server

context = {
    "text": "The Model Context Protocol enables structured communication between AI models and external systems."
}

result = server.handle_request("word_count", context)

print("Word count result:", result)
