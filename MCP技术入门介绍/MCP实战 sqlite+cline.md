# 1 uv配置环境

个人使用

~~~
uv venv
.venv\Scripts\activate
~~~

```
[project]
name = "mcpdemo"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "mcp>=1.6.0",
    "mysql>=0.0.3",
    "mysql-connector>=2.2.9",
    "selenium>=4.31.0",
    "tinydb>=4.8.2",
]

```

# 2 服务端

参考

[面向客户端开发者 - MCP 中文文档](https://mcp-docs.cn/quickstart/client)

[modelcontextprotocol/servers: Model Context Protocol Servers](https://github.com/modelcontextprotocol/servers)

[一起来玩mcp_server_sqlite，让AI帮你做增删改查！！ - mingupupup - 博客园](https://www.cnblogs.com/mingupupu/p/18773859)

[高手必看，Cline与MCP集成指南 - 知乎](https://zhuanlan.zhihu.com/p/20222456593)

```python
from mcp.server import FastMCP
from selenium.webdriver import Remote, ChromeOptions # These seem unused now, but kept
from selenium.webdriver.chromium.remote_connection import ChromiumRemoteConnection # Unused
from selenium.webdriver.common.by import By # Unused
from selenium.webdriver.support.ui import WebDriverWait # Unused
from selenium.webdriver.support import expected_conditions as EC # Unused
import time
import sqlite3
mcp = FastMCP("server")
DATABASE_FILE = "test.db"

def setup_database():
    """Creates the database file and the users table if they don't exist."""
    print(f"Setting up database at: {DATABASE_FILE}")
    try:
        # Connect (creates the file if it doesn't exist)
        with sqlite3.connect(DATABASE_FILE) as conn:
            cursor = conn.cursor()
            # Create table with an auto-incrementing ID and unique name constraint
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    age INTEGER,
                    address TEXT
                )
            """)
            conn.commit() # Commit table creation
            print("Database table 'users' checked/created successfully.")
    except sqlite3.Error as e:
        print(f"!!! Database setup error: {e}")
        # Depending on severity, you might want to exit or raise the exception
        raise # Re-raise the exception to stop the script if setup fails


@mcp.tool()
def get_user_info(name: str):
    """
    根据用户名查询用户信息 (使用 SQLite)

    Args:
        name (str): 用户名

    Returns:
        dict: 用户信息，包含id, name, age, address字段
        str: 如果未找到用户或发生错误
    """
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            # Use Row factory for dictionary-like access
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            sql = "SELECT id, name, age, address FROM users WHERE name = ?"
            cursor.execute(sql, (name,)) # Use tuple for placeholder
            result = cursor.fetchone()

            if result:
                return dict(result) # Convert sqlite3.Row to dict
            else:
                return f"User '{name}' not found."
    except sqlite3.Error as e:
        print(f"SQLite get_user_info error: {e}")
        return f"Database query error: {str(e)}"
    except Exception as e:
        print(f"Unexpected get_user_info error: {e}")
        return f"An unexpected error occurred: {str(e)}"

@mcp.tool()
def add_user(name: str, age: int, address: str):
    """
    创建新用户 (使用 SQLite)

    Args:
        name (str): 用户名
        age (int): 年龄
        address (str): 地址

    Returns:
        str: 操作结果消息
    """
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            cursor = conn.cursor()
            sql = "INSERT INTO users (name, age, address) VALUES (?, ?, ?)"
            cursor.execute(sql, (name, age, address))
            # Get the id of the inserted row
            user_id = cursor.lastrowid
            conn.commit() # Essential: commit the transaction
            return f"User '{name}' added successfully with ID {user_id}."
    except sqlite3.IntegrityError:
        # This error specifically occurs if the UNIQUE constraint on 'name' is violated
        print(f"SQLite add_user error: User '{name}' already exists.")
        return f"Error: User '{name}' already exists."
    except sqlite3.Error as e:
        print(f"SQLite add_user database error: {e}")
        return f"Database insert error: {str(e)}"
    except Exception as e:
        print(f"Unexpected add_user error: {e}")
        return f"An unexpected error occurred: {str(e)}"

if __name__ == "__main__":
    # --- Setup the database before starting the server ---
    setup_database()

    # --- Start the MCP server ---
    print("Starting MCP server...")
    mcp.run()
```

# 3 客户端Cline

这里使用CLine插件，配置Server

要注意，数据库目前mysql或者sqlite是支持的

安装一个能用大模型的客户端，且支持MCP

这里选择cline作为实践



配置mcp协议

```
{
  "mcpServers": {
  "sqlite": {
    "command": "uv",
    "args": [
      "--directory",
      "C:\\Users\\Chen\\Documents\\codes\\Github\\LLM\\MCP技术入门介绍\\MCP",
      "run",
      "McpServer.py",
      "--db-path",
      "~/test.db"
    ]
  }
}
}
```



![image-20250423221046390](C:\Users\Chen\Documents\codes\Github\LLM\MCP技术入门介绍\MCP\assets\image-20250423221046390.png)

# 4 MCP 效果

## 简单的添加和查询

![image-20250423221126949](C:\Users\Chen\Documents\codes\Github\LLM\MCP技术入门介绍\MCP\assets\image-20250423221126949.png)

## 让大模型知道从数据库搜数据

![image-20250423221213425](C:\Users\Chen\Documents\codes\Github\LLM\MCP技术入门介绍\MCP\assets\image-20250423221213425.png)