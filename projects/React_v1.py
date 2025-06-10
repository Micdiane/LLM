import os
import re
from dotenv import load_dotenv
import requests
import json

# 加载环境变量
load_dotenv()

class DeepSeekLLM:
    """DeepSeek LLM 客户端"""
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        
    def generate(self, prompt, max_tokens=1000, temperature=0.7):
        """生成文本"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error: {str(e)}"

class ReActAgent:
    """ReAct 推理代理"""
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {tool["name"]: tool for tool in tools}
        
    def run(self, question, max_iterations=5):
        """运行 ReAct 推理循环"""
        print(f"Question: {question}\n")
        
        # ReAct 提示模板
        prompt = f"""You are a helpful assistant that can use tools to answer questions.

Available tools:
{self._format_tools()}

Use the following format:
Thought: [your reasoning about what to do next]
Action: [tool name]
Action Input: [input to the tool]
Observation: [result from the tool]
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: [your final answer]

Question: {question}

Let's think step by step:
"""
        
        conversation = prompt
        
        for i in range(max_iterations):
            # 生成下一步
            response = self.llm.generate(conversation)
            print(f"LLM Response:\n{response}\n")
            
            # 解析响应
            if "Final Answer:" in response:
                # 提取最终答案
                final_answer = response.split("Final Answer:")[-1].strip()
                print(f"Final Answer: {final_answer}")
                return final_answer
            
            # 查找动作
            action_match = re.search(r"Action:\s*(.+)", response)
            action_input_match = re.search(r"Action Input:\s*(.+)", response)
            
            if action_match and action_input_match:
                action = action_match.group(1).strip()
                action_input = action_input_match.group(1).strip()
                
                print(f"Action: {action}")
                print(f"Action Input: {action_input}")
                
                # 执行工具
                if action in self.tools:
                    try:
                        observation = self.tools[action]["func"](action_input)
                        print(f"Observation: {observation}\n")
                        
                        # 更新对话
                        conversation += f"\n{response}\nObservation: {observation}\n"
                    except Exception as e:
                        observation = f"Error executing tool: {str(e)}"
                        print(f"Observation: {observation}\n")
                        conversation += f"\n{response}\nObservation: {observation}\n"
                else:
                    observation = f"Unknown tool: {action}"
                    print(f"Observation: {observation}\n")
                    conversation += f"\n{response}\nObservation: {observation}\n"
            else:
                # 如果没有找到动作，继续对话
                conversation += f"\n{response}\n"
        
        return "Maximum iterations reached without finding a final answer."
    
    def _format_tools(self):
        """格式化工具描述"""
        tool_descriptions = []
        for tool in self.tools.values():
            tool_descriptions.append(f"- {tool['name']}: {tool['description']}")
        return "\n".join(tool_descriptions)

# 定义工具
def get_capital(country_name):
    """获取国家的首都"""
    capital_map = {
        "France": "Paris",
        "Germany": "Berlin",
        "Italy": "Rome",
        "Spain": "Madrid",
        "China": "Beijing",
        "Japan": "Tokyo",
        "UK": "London",
        "USA": "Washington D.C."
    }
    if country_name in capital_map:
        return capital_map[country_name]
    else:
        return f"I don't know the capital of {country_name}."

def calculate(expression):
    """计算数学表达式"""
    try:
        # 简单的安全计算
        allowed_chars = set('0123456789+-*/.() ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return str(result)
        else:
            return "Invalid expression"
    except Exception as e:
        return f"Error: {str(e)}"

# 主程序
if __name__ == "__main__":
    # 初始化 LLM
    api_key = os.getenv("ds_api")
    if not api_key:
        print("请在 .env 文件中设置 ds_api")
        exit(1)
    
    llm = DeepSeekLLM(api_key)
    
    # 定义工具
    tools = [
        {
            "name": "get_capital",
            "description": "Get the capital city of a country. Input should be a country name.",
            "func": get_capital
        },
        {
            "name": "calculate",
            "description": "Calculate a mathematical expression. Input should be a valid math expression.",
            "func": calculate
        }
    ]
    
    # 创建 ReAct 代理
    agent = ReActAgent(llm, tools)
    
    # 测试问题
    questions = [
        "What is the capital of Germany and what is 12315324214 * 1145141919810?"
        ""
    ]
    
    for question in questions:
        print("=" * 50)
        result = agent.run(question)
        print(f"Final Result: {result}")
        print("=" * 50)
        print()
