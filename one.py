from tavily import TavilyClient
from openai import OpenAI
import requests 
import os
import re
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 输出格式要求:
你的每次回复必须严格遵循以下格式，包含一对Thought和Action：

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 重要提示:
- 每次只输出一对Thought-Action
- Action必须在同一行，不要换行
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束

请开始吧！
"""

def get_weather(city:str):
	url=f"https://wttr.in/{city}?format=j1"
	try : 
		response=requests.get(url)
		response.raise_for_status() 
		data=response.json()

		current_condition = data['current_condition'][0]
		weather_desc = current_condition['weatherDesc'][0]['value']
		temp_c = current_condition['temp_C']
		return f"{city}的天气是{weather_desc}，温度是{temp_c}摄氏度"
	except requests.exceptions.RequestException as e:
		return f"获取{city}的天气失败：{e}"
	except (KeyError, IndexError) as e:
		return f"{city}的天气数据格式错误：{e}"
	
def get_attraction(city:str,weather:str=""):
	api_key=os.environ.get("TAVILY_API_KEY")
	if not api_key:
		return "错误:未配置TAVILY_API_KEY环境变量。"
	tavily = TavilyClient(api_key=api_key)
	query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"
	try:
		
		response = tavily.search(query=query, search_depth="basic", include_answer=True)
		
		
		if response.get("answer"):
			return response["answer"]
		
		
		formatted_results = []
		for result in response.get("results", []):
			formatted_results.append(f"- {result['title']}: {result['content']}")
		
		if not formatted_results:
			return "抱歉，没有找到相关的旅游景点推荐。"

		return "根据搜索，为您找到以下信息:\n" + "\n".join(formatted_results)

	except Exception as e:
		return f"错误:执行Tavily搜索时出现问题 - {e}"
	
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}
class DeepSeekClient:
	def __init__(self, model: str,api_key: str,base_url: str):
		self.model=model
		self.client = OpenAI(api_key=api_key, base_url=base_url)
	def generate(self, prompt: str, system_prompt: str=""):
		print("正在调用DeepSeek模型...")
		try:
			response = self.client.chat.completions.create(
				model=self.model,
				messages=[
					{"role": "system", "content": system_prompt},
					{"role": "user", "content": prompt}
				],
				max_tokens=1024,
				temperature=0.7
			)
			return response.choices[0].message.content
		except Exception as e:
			return f"调用DeepSeek模型时出现问题 - {e}"
		

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"
MODEL_ID = "deepseek-chat"
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

if not API_KEY:
    print("错误: 未配置DEEPSEEK_API_KEY环境变量")
    print("请设置环境变量: export DEEPSEEK_API_KEY=your_api_key")
    exit(1)

if not TAVILY_API_KEY:
    print("错误: 未配置TAVILY_API_KEY环境变量")
    print("请设置环境变量: export TAVILY_API_KEY=your_api_key")
    exit(1)

llm = DeepSeekClient(
    model=MODEL_ID,
    api_key=API_KEY,
    base_url=BASE_URL
)
user_prompt = "你好，请帮我查询一下今天广州的天气，然后根据天气推荐一个合适的旅游景点。"
prompt_history = [f"用户请求: {user_prompt}"]
print(f"用户输入: {user_prompt}\n" + "="*40)

for i in range(5):
	print(f"--- 循环 {i+1} ---\n")

	full_prompt = "\n".join(prompt_history)
	llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
	if not llm_output:
		print("模型返回空值，跳过本轮")
		continue
	match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output, re.DOTALL)
	if match:
		truncated = match.group(1).strip()
		if llm_output and truncated != llm_output.strip():
			llm_output = truncated
			print("已截断多余的 Thought-Action 对")
	print(f"模型输出:\n{llm_output}\n")

	prompt_history.append(llm_output if llm_output!=None else "")

	# 3.3. 解析并执行行动
	action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
	if not action_match:
		observation = "错误: 未能解析到 Action 字段。请确保你的回复严格遵循 'Thought: ... Action: ...' 的格式。"
		observation_str = f"Observation: {observation}"
		print(f"{observation_str}\n" + "="*40)
		prompt_history.append(observation_str)
		continue
	action_str = action_match.group(1).strip()
	if action_str.startswith("Finish"):
		finish_match = re.match(r"Finish\[(.*)\]", action_str)
		if finish_match:
			final_answer = finish_match.group(1)
			print(f"任务完成，最终答案: {final_answer}")
			break
		else:
			observation = "错误: Finish 格式不正确，应该是 Finish[你的答案]"
			observation_str = f"Observation: {observation}"
			print(f"{observation_str}\n" + "="*40)
			prompt_history.append(observation_str)
			continue
	
	tool_name_match = re.search(r"(\w+)\(", action_str)
	if not tool_name_match:
		observation = "错误: 无法解析工具名称"
		observation_str = f"Observation: {observation}"
		print(f"{observation_str}\n" + "="*40)
		prompt_history.append(observation_str)
		continue
	tool_name = tool_name_match.group(1)
	
	args_str_match = re.search(r"\((.*)\)", action_str)
	if not args_str_match:
		observation = "错误: 无法解析工具参数"
		observation_str = f"Observation: {observation}"
		print(f"{observation_str}\n" + "="*40)
		prompt_history.append(observation_str)
		continue
	args_str = args_str_match.group(1)
	
	kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))
	
	if tool_name in available_tools:
		observation = available_tools[tool_name](**kwargs)
		
	else:
		observation = f"错误: 未找到工具 {tool_name}"

	observation_str = f"Observation: {observation}"
	print(f"{observation_str}\n" + "="*40)
	prompt_history.append(observation_str)