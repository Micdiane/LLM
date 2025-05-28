# Function Calling

大模型通过外部函数实现

底层原理：

[大模型工具调用(function call)原理及实现 - 知乎](https://zhuanlan.zhihu.com/p/663770472)

模型如何获得FC的能力——指令微调——教会大模型格式化输出

简单来说，**Function Calling 是一种让大型语言模型 (LLM) 能够更可靠地与外部工具或 API 进行交互的机制。** 它允许 LLM 不仅仅是生成文本，还能识别出何时需要调用一个外部函数（比如查询天气、发送邮件、访问数据库等），并以结构化的格式（通常是 JSON）返回需要调用哪个函数以及调用时所需的参数。

**解决的问题：**

标准的大语言模型本质上是文本生成器。它们可以根据提示生成连贯、相关的文本，但它们本身无法执行现实世界的操作或访问实时、私有的信息。例如：

*   你问 LLM：“明天北京天气怎么样？” 它可能根据训练数据中的知识回答一个大概的天气情况，但这并不是实时的。
*   你让 LLM：“帮我给我同事发邮件，告诉他会议推迟到下午 3 点。” LLM 只能生成邮件的文本内容，但它无法 *真正地* 发送邮件。

**Function Calling 的工作原理：**

Function Calling 机制通过以下步骤解决了这个问题：

1.  **定义可用函数 (开发者提供):** 开发者在调用 LLM API 时，除了提供用户输入的 prompt 外，还需要提供一个或多个“可用函数”的描述列表。这个描述通常包括：
    *   **函数名称 (Function Name):** 比如 `get_current_weather` 或 `send_email`。
    *   **函数描述 (Description):** 清晰地说明这个函数的作用，帮助 LLM 理解何时应该使用它。例如，“获取指定地点的当前天气信息”。
    *   **函数参数 (Parameters):** 定义该函数需要哪些输入参数，包括参数名称、类型（如 string, integer, enum）、描述以及是否必需。例如，`get_current_weather` 需要一个 `location` (string, 必需) 和一个可选的 `unit` (enum: 'celsius' 或 'fahrenheit')。

2.  **LLM 分析与决策:** 当 LLM 接收到用户的请求（比如 “明天北京天气怎么样？”）以及可用的函数列表时，它会：
    *   **理解用户意图:** 分析用户的请求是想获取实时天气信息。
    *   **匹配函数:** 在提供的函数列表中查找最适合满足用户意图的函数（找到了 `get_current_weather`）。
    *   **提取参数:** 从用户的请求中提取调用该函数所需的参数（提取出 `location="北京"`，可能还需要判断时间是“明天”）。

3.  **LLM 返回结构化指令:** **关键点来了：** LLM 不会直接执行这个函数。相反，它会返回一个结构化的数据（通常是 JSON 对象），表明它决定调用哪个函数以及需要传入哪些参数。
    
    *   例如，LLM 的返回可能类似于：
        ```json
        {
          "function_call": {
            "name": "get_current_weather",
            "arguments": "{\"location\": \"北京\", \"date\": \"tomorrow\"}"
          }
        }
        ```
        *(注意：参数 `arguments` 通常是一个 JSON 字符串，需要再次解析)*
    
4.  **应用程序执行函数 (开发者负责):** 你的应用程序代码接收到 LLM 返回的这个 JSON 指令。
    *   **解析指令:** 代码解析出要调用的函数名 (`get_current_weather`) 和参数 (`{"location": "北京", "date": "tomorrow"}`)。
    *   **调用实际函数:** 你的代码根据这些信息，**真正地**调用你本地定义的 `get_current_weather` 函数（这个函数内部可能会去请求一个真实的天气 API）。

5.  **(可选) 将结果返回给 LLM:** 你的应用程序执行完函数后，得到了结果（比如 “明天北京天气：晴，15-25 摄氏度”）。你可以选择将这个结果再次发送给 LLM。

6.  **LLM 生成最终回复:** 如果你将函数执行结果返回给了 LLM，它可以基于这个具体、实时的信息，生成一个更自然、更准确的最终回复给用户。例如：“明天的北京天气预报是晴天，气温在 15 到 25 摄氏度之间。”

**Function Calling 的主要优势：**

1.  **扩展 LLM 能力:** 让 LLM 能够“连接”外部世界，利用各种工具、API 和数据库。
2.  **结构化数据提取:** 可以非常可靠地从自然语言中提取出结构化的信息（比如 JSON）。
3.  **可靠的输出格式:** 保证 LLM 的输出是可预测的、机器可读的格式，方便程序调用。
4.  **构建智能代理 (Agent):** 是实现更复杂的 LLM Agent (如 ReAct 框架中的工具使用) 的基础，让 LLM 可以规划并执行一系列涉及外部工具的操作。
5.  **提高准确性与时效性:** 通过调用外部 API 获取实时、准确的数据，减少 LLM 的“幻觉” (Hallucination)。

MCP减少重复造轮子

# MCP

[MCP 简介 - MCP 中文文档](https://mcp-docs.cn/introduction)

MCP 是一个开放协议，它为应用程序向 LLM 提供上下文的方式进行了标准化。你可以将 MCP 想象成 AI 应用程序的 USB-C 接口。就像 USB-C 为设备连接各种外设和配件提供了标准化的方式一样，MCP 为 AI 模型连接各种数据源和工具提供了标准化的接口。

![image-20250423210254231](C:\Users\Chen\Documents\codes\Github\LLM\MCP技术入门介绍\assets\image-20250423210254231.png)

发布工具和使用工具

[什么是MCP｜工作原理｜如何使用MCP｜图解MCP - 知乎](https://zhuanlan.zhihu.com/p/32975857666)

[MCP详解及手把手实战 | 华为开发者联盟](https://developer.huawei.com/consumer/cn/blog/topic/03180540268583022)

[punkpeye/awesome-mcp-servers: A collection of MCP servers.](https://github.com/punkpeye/awesome-mcp-servers)

[百炼控制台](https://bailian.console.aliyun.com/?tab=mcp#/mcp-market)

[jlowin/fastmcp: 🚀 The fast, Pythonic way to build MCP servers and clients](https://github.com/jlowin/fastmcp)

不需要用户方格式严格对齐

大模型会自己对齐，还会给样例