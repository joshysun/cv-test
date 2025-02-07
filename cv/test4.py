# System message
import operator, os, getpass
from typing import TypedDict, List, Dict, Annotated

from IPython.core.display import Image
from IPython.core.display_functions import display
from langchain_core.messages import SystemMessage, AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import START
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


# dotenv.load_dotenv()

client = ChatOpenAI(model="gpt-4o", temperature=0)


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
_set_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "cv-test"


# Define state types
class AgentState(TypedDict):
    messages: List[BaseMessage]
    stage: str
    education: Dict
    work_experience: Dict
    skills: Dict

# 目前沒用到這個prompt了
def get_system_prompt():
    return f""""# 履歷協作專家
語言：繁體中文
模式：三階段引導式對話
核心目標：5分鐘內完成高精度結構化履歷

# 互動系統設計

分段流程蒐集下方的履歷欄位資訊

第一階段「學歷背景」→ 第二階段「工作經歷」→ 第三階段「專業技能」

學歷背景包含：大學、研究所，畢業 / 肄業，起始、結束年月份，若user沒有特別說是大學或研究所，一律默認是大學畢業
工作經歷包含：公司名稱，起始、結束年月份，工作內容(配對到專長)
專業技能包含：語言，擅長工具...

需在適當的時機詢問求職條件，並將蒐集到的資訊填入對應的欄位中

每階段開始時顯示進度百分比（25% → 50% → 75%）

雙向確認機制

每次回答後即時顯示「中文鍵值對」確認，並且提供一小段符合履歷敘述的句子

公司名稱: 台灣人工智慧科技, 職務名稱: 資深工程師
異常數據自動標註 ❌ 並提供修正指示

輔助功能

自動轉換簡稱（例：「臺大」→「國立台灣大學」）

根據日期智能判斷就職狀態（在職/離職）

自由格式回答自動映射到對應欄位

# 對話流程規範
步驟1：系統初始化

開場顯示互動規則看板

包含即時預覽範例與修改指令說明

步驟2：階段性資料收集

每階段最多詢問4個關聯問題

重要欄位（如日期）自動格式驗證

步驟3：動態補齊機制

未完成欄位智能推薦補問策略

允許跨階段回溯修正資料

步驟4：最終輸出

根據欄位及對應的值，完成一份簡單的履歷，並且輸出

# 對話腳本模板
[教育階段]
「請分享您的最高學歷：學校全稱與就讀科系是？
（例：國立成功大學 / 電機工程學系）」

# 用戶回答後顯示：
✅ 已儲存：
┌ 學校名稱 → 國立成功大學
└ 科系名稱 → 電機工程學系

[異常處理範例]
# 檢查到時間矛盾：
「畢業年月：2023-09」 早於「入學年月：2025-06」
請確認是否需調整日期或說明特殊情況？
  如果使用者偏題，沒有回覆履歷相關問題，請將對話導回蒐集履歷條件

[完成通知]
# 履歷完整度 92% | 剩餘可選填欄位：2項

[預覽結構化數據]
輸入「修改 職稱」或「確認提交」以繼續

# 效能強化模組

情境感知：自動識別「同上間公司」等上下文參照
錯誤緩衝：允許3次格式錯誤後啟動智能修正建議

# 履歷欄位
```
//學校名稱  school_name: str
//學歷    education_levels: List[str]
//科系名稱  department_name: str
//就學期間-起始   school_start_date: str
//就學期間-結束   school_end_date: str
//就學狀態  educational_states: List[str]
//公司名稱  company_name: str
//職務名稱  job_category: str
//任職期間-起始   term_start_date: str
//任職期間-結束   term_end_date: str
//希望地點  hope_work_cities: List[str]
//希望職稱  hope_job_title: str
//專長名稱  good_at_skills_name: List[str]
//外文種類  good_at_languages: List[str]
```


*請用繁體中文進行對話流程*
*盡量用最精簡的句子提問、回答，不要有過多的冗言贅詞*"""

def create_education_system_prompt():
    return """你是一位履歷協作專家，現在是「學歷背景」收集的階段。
    請從用戶的回答中擷取以下資訊：
    - 學校名稱  school_name: str
    - 最高學歷  education_levels: List[str]
    - 科系名稱  department_name: str
    - 就學期間-起始   school_start_date: str
    - 就學期間-結束   school_end_date: str
    - 就學狀態  educational_states: List[str]
    請以對話方式引導用戶提供完整資訊。"""


def create_work_system_prompt():
    return """你是一位履歷協作專家，現在是第二階段「工作經歷」收集。
    請從用戶的回答中擷取以下資訊：
    - 公司名稱  company_name: str
    - 職稱    job_category: str
    - 任職期間-起始  term_start_date: str
    - 任職期間-結束  term_end_date: str
    - 希望地點  hope_work_cities: List[str]
    - 希望職稱  hope_job_title: str
    請以對話方式引導用戶提供完整資訊。"""


def create_skills_system_prompt():
    return """你是一位履歷協作專家，現在是第三階段「專業技能」收集。
    請從用戶的回答中擷取以下資訊：
    - 專長名稱  good_at_skills_name: List[str]
    - 外文種類  good_at_languages: List[str]
    請以對話方式引導用戶提供完整資訊。"""


def get_llm_response(messages: List[BaseMessage]) -> str:
    """Helper function to get response from GPT-4"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": msg.type, "content": msg.content} for msg in messages],
        temperature=1,
    )
    return response.choices[0].message.content


def start(state: AgentState) -> AgentState:
    """Initial function to start the conversation"""
    system_message = """你是一位履歷協作專家，將在5分鐘內完成高精度結構化履歷。
    請以友善的語氣開始對話，說明接下來會分三個階段收集資訊：
    1. 學歷背景
    2. 工作經歷
    3. 專業技能
    請開始引導用戶進入第一階段。
    *請用繁體中文進行對話流程*
    *盡量用最精簡的句子提問、回答，不要有過多的冗言贅詞*
    """

    messages = [
        SystemMessage(content=system_message),
        *state["messages"]
    ]

    response = get_llm_response(messages)
    state["messages"].append(AIMessage(content=response))
    state["stage"] = "education"
    return state


def collect_education(state: AgentState, school_name: str, education_levels: List[str], department_name: str, school_start_date: str, school_end_date: str,
                      educational_states: List[str]) -> AgentState:
    """Collect education information"""
    messages = [
        SystemMessage(content=create_education_system_prompt()),
        *state["messages"]
    ]

    response = get_llm_response(messages)
    state["messages"].append(AIMessage(content=response))

    # Extract education info from the conversation
    # In a real implementation, you might want to use more sophisticated parsing
    if len(state["messages"]) >= 4:  # Assuming we have enough context
        state["education"] = {
            "status": "complete"
            # Add actual parsing logic here
        }
        state["stage"] = "work"

    return state


def collect_work_experience(state: AgentState, company_name: str, job_category: str, term_start_date: str, term_end_date: str,
                            hope_work_cities: list[str], hope_job_title: str) -> AgentState:
    """Collect work experience information"""
    messages = [
        SystemMessage(content=create_work_system_prompt()),
        *state["messages"]
    ]

    response = get_llm_response(messages)
    state["messages"].append(AIMessage(content=response))

    # Extract work experience info from the conversation
    if len(state["messages"]) >= 6:  # Assuming we have enough context
        state["work_experience"] = {
            "status": "complete"
            # Add actual parsing logic here
        }
        state["stage"] = "skills"

    return state


def collect_skills(state: AgentState, good_at_skills_name: list[str], good_at_languages: list[str]) -> AgentState:
    """Collect professional skills information"""
    messages = [
        SystemMessage(content=create_skills_system_prompt()),
        *state["messages"]
    ]

    response = get_llm_response(messages)
    state["messages"].append(AIMessage(content=response))

    # Extract skills info from the conversation
    if len(state["messages"]) >= 12:  # Assuming we have enough context
        state["skills"] = {
            "status": "complete"
            # Add actual parsing logic here
        }
        state["stage"] = "complete"

    return state


def should_continue(state: AgentState) -> bool:
    """Determine if we should continue collecting information"""
    return state["stage"] != "complete"


def get_next_stage(state: AgentState) -> Annotated[str, operator.eq]:
    """Determine which stage to execute next"""
    return state["stage"]

sys_msg = SystemMessage(content="""你是一位履歷協作專家，將在5分鐘內完成高精度結構化履歷；每次回答後即時顯示「中文鍵值對」確認，並且提供一小段符合履歷敘述的句子。
每次回答後即時顯示「中文鍵值對」確認，並且提供一小段符合履歷敘述的句子。
公司名稱: 台灣人工智慧科技, 職務名稱: 資深工程師
異常數據自動標註 ❌ 並提供修正指示

輔助功能

自動轉換簡稱（例：「臺大」→「國立台灣大學」）

根據日期智能判斷就職狀態（在職/離職）

自由格式回答自動映射到對應欄位
# 對話流程規範
- 階段性資料收集
每階段最多詢問4個關聯問題
重要欄位（如日期）自動格式驗證
- 動態補齊機制
未完成欄位智能推薦補問策略
允許跨階段回溯修正資料
- 最終輸出
根據欄位及對應的值，完成一份簡單的履歷，並且輸出
# 對話腳本模板
[教育階段]
「請分享您的最高學歷：學校全稱與就讀科系是？
（例：國立成功大學 / 電機工程學系）」

# 用戶回答後顯示：
✅ 已儲存：
┌ 學校名稱 → 國立成功大學
└ 科系名稱 → 電機工程學系

[異常處理範例]
# 檢查到時間矛盾：
「畢業年月：2023-09」 早於「入學年月：2025-06」
請確認是否需調整日期或說明特殊情況？
如果使用者偏題，沒有回覆履歷相關問題，請將對話導回蒐集履歷條件
# 履歷欄位
```
//學校名稱  school_name: str
//學歷    education_levels: List[str]
//科系名稱  department_name: str
//就學期間-起始   school_start_date: str
//就學期間-結束   school_end_date: str
//就學狀態  educational_states: List[str]
//公司名稱  company_name: str
//職務名稱  job_category: str
//任職期間-起始   term_start_date: str
//任職期間-結束   term_end_date: str
//希望地點  hope_work_cities: List[str]
//希望職稱  hope_job_title: str
//專長名稱  good_at_skills_name: List[str]
//外文種類  good_at_languages: List[str]
```
*請用繁體中文進行對話流程*
*盡量用最精簡的句子提問、回答，不要有過多的冗言贅詞*""")

tools = [get_llm_response, start, collect_education, collect_work_experience, collect_skills, should_continue, get_next_stage]
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")
react_graph = builder.compile()

# Show
# display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))

# messages = [HumanMessage(content="我想找工作")]
# messages = react_graph.invoke({"messages": messages})

# for m in messages['messages']:
#     m.pretty_print()


def interactive_resume_builder():
    """交互式履歷建立工具"""
    print("開始寫履歷！")
    print("請開始輸入您的履歷資訊。輸入 'exit' 結束程式。")
    print("""嗨！我是您的履歷協作專家，歡迎一起完成高精度履歷！
我們將從學歷開始，依次完成每個階段資料蒐集。

請分享您的最高學歷：學校全稱與就讀科系是？
（例：國立成功大學 / 電機工程學系）。
輸入 'exit' 結束程式。""")

    messages = [HumanMessage(content="hihi")]

    while True:
        # 執行工作流程
        result = react_graph.invoke({"messages": messages})

        # print AI的最後一個回覆
        last_ai_message = [m for m in result['messages'] if isinstance(m, AIMessage)][-1]
        print("\n[AI]:", last_ai_message.content)

        # 用戶輸入
        user_input = input("\n[你]：").strip()

        # 檢查是否要退出
        if user_input.lower() == 'exit':
            print("感謝使用履歷協作專家，再見！")
            break

        # 將用戶輸入加入message
        messages.append(HumanMessage(content=user_input))


if __name__ == "__main__":
    interactive_resume_builder()