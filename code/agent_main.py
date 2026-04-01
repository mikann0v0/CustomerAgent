import json
import re
from typing import Any, Dict, List, Optional
from pathlib import Path

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
from langgraph.graph import StateGraph, END

from state import AgentState
from config import (
    CHAT_MODEL, EMBEDDING_MODEL, VISION_MODEL,
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
    SILICON_API_KEY, SILICON_BASE_URL
)
from kb import KnowledgeBaseManager

class MultimodalCustomerAgent:
    def __init__(
        self,
        docs_dir: Optional[str] = None,
        pics_dir: Optional[str] = None,
        persist_dir: Optional[str] = None,
    ):
        base_dir = Path(__file__).resolve().parents[1]
        self.docs_dir = Path(docs_dir) if docs_dir else base_dir / "docs"
        self.pics_dir = Path(pics_dir) if pics_dir else self.docs_dir / "pics"
        self.persist_dir = Path(persist_dir) if persist_dir else Path(__file__).resolve().parent / "qdrant_db"

        self.llm_model = CHAT_MODEL
        self.llm_temperature = 0.1

        # Use OpenRouter for Chat if Key is provided, else fallback to SiliconFlow chat models
        if OPENROUTER_API_KEY and OPENROUTER_API_KEY != "TO_BE_FILLED_LATER":
            self.llm = ChatOpenAI(
                model=self.llm_model,
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL,
                temperature=self.llm_temperature
            )
        else:
            self.llm_model = "Qwen/Qwen2.5-7B-Instruct"
            self.llm = ChatOpenAI(
                model=self.llm_model,
                api_key=SILICON_API_KEY,
                base_url=SILICON_BASE_URL,
                temperature=self.llm_temperature
            )

        self.vision_model = VISION_MODEL
        self.vision_client = OpenAI(api_key=SILICON_API_KEY, base_url=SILICON_BASE_URL)

        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=SILICON_API_KEY,
            base_url=SILICON_BASE_URL,
        )

        self.kb = KnowledgeBaseManager(
            docs_dir=self.docs_dir,
            pics_dir=self.pics_dir,
            persist_dir=self.persist_dir,
            embeddings=self.embeddings,
            vision_client=self.vision_client,
            vision_model=self.vision_model
        )

        # Initialize the LangChain Agent
        self.agent_executor = self._create_agent()

    def build_knowledge_base(self) -> Dict[str, int]:
        return self.kb.build_knowledge_base()

    def _create_agent(self) -> Any:
        @tool
        def search_manual(query: str) -> str:
            """用于回答产品说明书、使用方法、部件说明、故障排查、操作步骤等技术问题时，从向量库检索说明书信息。"""
            pass # Dummy implementation for binding

        llm_with_tech_tools = self.llm.bind_tools([search_manual])

        def supervisor_node(state: AgentState):
            supervisor_prompt = """你是智能客服系统的主管调度员(Supervisor)。
你的唯一任务是分析用户的最新输入，并将其分配派发给最合适的专家团队。

可选的专家团队(next_agent)及对应职责：
- "tech_agent": 专门处理产品说明书、使用方法、功能咨询、故障排查、操作步骤等技术问题。（例如：“如何启动发电机？”、“空调清洗滤网”）
- "after_sales_agent": 专门处理退货、换货、维修政策、发票、商品损坏索赔等售后问题。（例如：“支持7天无理由吗”、“发票开错了”）
- "logistics_agent": 专门处理物流进度、少发、漏发、没收到货、快递员态度等订单物流问题。（例如：“快递丢了”、“少发了一件”）
- "FINISH": 用户纯粹宣泄情绪、打招呼或者表达感谢，不需要专家介入。

必须且只能返回合法的 JSON 格式数据，不允许输出多余的解释文本，格式如下：
{"next_agent": "选定的团队", "reason": "简短的一句话理由"}
"""
            user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
            if not user_messages:
                return {"next_agent": "tech_agent"}

            messages = [SystemMessage(content=supervisor_prompt), user_messages[-1]]

            try:
                response = self.llm.invoke(messages)
                content = response.content
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    content = match.group(0)
                parsed = json.loads(content)
                next_agent = parsed.get("next_agent", "tech_agent")

                valid_agents = ["tech_agent", "after_sales_agent", "logistics_agent", "FINISH"]
                if next_agent not in valid_agents:
                    next_agent = "tech_agent"

                print(f"Supervisor Routing -> {next_agent} (Reason: {parsed.get('reason', 'N/A')})")
                return {"next_agent": next_agent}
            except Exception as e:
                print(f"Supervisor Error: {e}, fallback to tech_agent")
                return {"next_agent": "tech_agent"}

        def tech_agent_node(state: AgentState):
            tech_prompt = """你是一名严谨的产品技术专家。
你的职责是解答用户关于产品使用、说明书、故障排查等方面的问题。
遇到具体技术问题时，你必须调用 `search_manual` 工具检索知识库。
并在最终回答中归纳检索到的步骤。如果检索结果中有提到图片总结，必须在对应位置加上标记，如 (如图：[图片ID])。"""
            messages = [SystemMessage(content=tech_prompt)] + state["messages"]
            response = llm_with_tech_tools.invoke(messages)
            return {"messages": [response]}

        def after_sales_node(state: AgentState):
            prompt = """你是一名富有同理心的售后专员。
处理退换货、退款、发票、商品损坏索赔、维修流程等售后服务相关的用户提问。
请直接回复：向用户索要订单号和相关照片凭证，表示会尽快转接人工售后处理。语气要温和抱歉。"""
            messages = [SystemMessage(content=prompt), state["messages"][-1]]
            response = self.llm.invoke(messages)
            return {"messages": [response]}

        def logistics_node(state: AgentState):
            prompt = """你是专业的订单物流跟进员。
处理物流进度查询、未收到货、少发漏发、快递丢失、快递员态度差等订单物流问题。
请直接回复：礼貌地让用户提供订单号和快递单号，并告知会马上为您核实物流状态或安排补发。"""
            messages = [SystemMessage(content=prompt), state["messages"][-1]]
            response = self.llm.invoke(messages)
            return {"messages": [response]}

        def fallback_node(state: AgentState):
            prompt = """你是一个有礼貌的智能客服。
用户并没有询问产品技术、物流或售后问题。可能只是打招呼、感谢或抱怨。
请给出简短的安抚或礼貌回复。如果用户抱怨，请安抚情绪并提示可以留下订单号转接人工。"""
            messages = [SystemMessage(content=prompt), state["messages"][-1]]
            response = self.llm.invoke(messages)
            return {"messages": [response]}

        def tool_node(state: AgentState):
            messages = state["messages"]
            last_message = messages[-1]
            tool_calls = getattr(last_message, "tool_calls", [])

            tool_messages = []
            new_images = []
            new_paths = []
            new_refs = []

            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                query = tool_call["args"].get("query", "")

                if tool_name == "search_manual":
                    res = self.kb.search(query, state.get("top_k", 8))
                else:
                    res = {"text": f"Unknown tool: {tool_name}", "image_list": [], "image_paths": [], "references": []}

                tool_messages.append(ToolMessage(content=res["text"], tool_call_id=tool_call["id"]))
                new_images.extend(res.get("image_list", []))
                new_paths.extend(res.get("image_paths", []))
                new_refs.extend(res.get("references", []))

            return {
                "messages": tool_messages,
                "image_list": new_images,
                "image_paths": new_paths,
                "references": new_refs
            }

        def should_continue_from_supervisor(state: AgentState):
            return state.get("next_agent", "tech_agent")

        def tech_should_continue(state: AgentState):
            messages = state["messages"]
            last_message = messages[-1]
            if not getattr(last_message, "tool_calls", None):
                return END
            return "tech_tools"

        workflow = StateGraph(AgentState)

        workflow.add_node("supervisor", supervisor_node)
        workflow.add_node("tech_agent", tech_agent_node)
        workflow.add_node("after_sales_agent", after_sales_node)
        workflow.add_node("logistics_agent", logistics_node)
        workflow.add_node("FINISH", fallback_node)
        workflow.add_node("tech_tools", tool_node)

        workflow.set_entry_point("supervisor")

        workflow.add_conditional_edges(
            "supervisor",
            should_continue_from_supervisor,
            {
                "tech_agent": "tech_agent",
                "after_sales_agent": "after_sales_agent",
                "logistics_agent": "logistics_agent",
                "FINISH": "FINISH"
            }
        )

        workflow.add_conditional_edges(
            "tech_agent",
            tech_should_continue,
            ["tech_tools", END]
        )
        workflow.add_edge("tech_tools", "tech_agent")

        workflow.add_edge("after_sales_agent", END)
        workflow.add_edge("logistics_agent", END)
        workflow.add_edge("FINISH", END)

        return workflow.compile()

    def analyze_and_answer(self, question: str, chat_history: Optional[List[BaseMessage]] = None, top_k: int = 8) -> Dict[str, Any]:
        """入口函数：传入问题和历史对话，调用 LangGraph 统一处理多 Agent 工作流"""
        try:
            messages = []
            if chat_history:
                messages.extend(chat_history)
            messages.append(HumanMessage(content=question))

            initial_state = {
                "messages": messages,
                "image_list": [],
                "image_paths": [],
                "references": [],
                "top_k": top_k,
                "next_agent": ""
            }

            response = self.agent_executor.invoke(initial_state)
            answer = response["messages"][-1].content

            images_raw = response.get("image_list", [])
            paths_raw = response.get("image_paths", [])

            unique_images = list(dict.fromkeys(images_raw))
            unique_paths = list(dict.fromkeys(paths_raw))

            image_map = dict(zip(unique_images, unique_paths))

            for img_id, img_path in image_map.items():
                if not img_path:
                    continue

                # 计算相对路径
                import os
                try:
                    rel_img_path = os.path.relpath(img_path, start=str(Path(__file__).resolve().parent))
                except Exception:
                    rel_img_path = img_path

                # 替换为 Markdown 格式，让图片能够正常显示
                markdown_img = f"![{img_id}]({rel_img_path})"

                # 预先清理可能已经错误替换成的嵌套格式，比如 ![![img_id](/path.jpg)](/path.jpg)
                # 使用非贪婪匹配去掉错误的嵌套
                # 首先处理常规文本中的 "(如图：img_id)" 这类情况，确保替换不带错误嵌套

                # 为避免重复替换，使用占位符
                placeholder = f"__IMG_PLACEHOLDER_{img_id}__"

                # 将各种情况统一替换为 placeholder
                answer = re.sub(rf"（如图：\s*{img_id}）", placeholder, answer)
                answer = re.sub(rf"\(如图：\s*{img_id}\)", placeholder, answer)
                answer = re.sub(rf"如图：\s*{img_id}", placeholder, answer)
                answer = re.sub(rf"如图:\s*{img_id}", placeholder, answer)
                answer = re.sub(rf"\[{img_id}\]", placeholder, answer)

                # 针对已经被替换过的、或以其他嵌套形式存在的，也可以先剔除掉嵌套的括号
                # 针对孤立的 img_id，确保它前面没有 ![，后面没有 ]，也没有被 placeholder 匹配到
                pattern = rf"(?<!\[)(?<!\!)(\b{img_id}\b)(?!\])(?!\()"
                answer = re.sub(pattern, placeholder, answer)

                # 最后将 placeholder 替换为标准的 Markdown 图片格式
                answer = answer.replace(placeholder, markdown_img)

            # 清理可能的错误嵌套 markdown 图片（如果是在前面步骤已经污染的数据）
            # 这个正则处理类似于 ![![img_id](/path.jpg)](/path.jpg) 的情况
            for img_id, img_path in image_map.items():
                if not img_path:
                    continue
                try:
                    rel_img_path = os.path.relpath(img_path, start=str(Path(__file__).resolve().parent))
                except Exception:
                    rel_img_path = img_path
                correct_markdown = f"![{img_id}]({rel_img_path})"
                # 如果发现回答中出现了类似 !![id](path) 或者 ![![id](path)](path)
                # 直接统一清理为 correct_markdown
                # 简单的处理：如果存在两次以上该图片的路径或id混合，强制替换
                # 最简单的方法是用正确的图片字符串完全覆盖掉异常的图片字符串
                wrong_pattern_1 = rf"!!\[{img_id}\]\({rel_img_path}\)\({rel_img_path}\)"
                answer = answer.replace(wrong_pattern_1, correct_markdown)

                wrong_pattern_2 = rf"!\[!\[{img_id}\]\([^)]+\)\]\([^)]+\)"
                answer = re.sub(wrong_pattern_2, correct_markdown, answer)

            # 清除那些已经在文本回答中展示过的图片ID
            final_image_list = []
            final_image_paths = []
            for img_id, img_path in image_map.items():
                try:
                    rel_img_path = os.path.relpath(img_path, start=str(Path(__file__).resolve().parent))
                except Exception:
                    rel_img_path = img_path
                # 判断 markdown 图片是否在回答中出现了
                correct_markdown = f"![{img_id}]({rel_img_path})"
                if correct_markdown not in answer:
                    final_image_list.append(img_id)
                    final_image_paths.append(rel_img_path)

            # 把未在正文中显示的图片，附带链接添加到回答的最后作为补充资料
            if final_image_list:
                answer += "\n\n**补充资料（相关图片）：**\n"
                for img_id, img_path in zip(final_image_list, final_image_paths):
                    answer += f"![{img_id}]({img_path})\n"

            return {
                "ret": answer,
                "image_list": final_image_list,
                "image_paths": final_image_paths,
                "references": response.get("references", []),
            }
        except Exception as e:
            print(f"Agent Execution Error: {e}")
            return {
                "ret": "抱歉，系统暂时出现异常，未能处理您的请求。请稍后再试或联系人工客服。",
                "image_list": [],
                "image_paths": [],
                "references": []
            }

if __name__ == "__main__":
    # Note: We commented out KnowledgeBaseManager init in __init__ to avoid building KB
    # For this test, we assume qdrant_db already has the necessary data
    # We need to re-enable kb init but without rebuilding it if it already exists
    # Actually, KnowledgeBaseManager init doesn't build, it just connects to Qdrant.
    agent = MultimodalCustomerAgent()

    test_questions = [
        "VR头显如何调节瞳距和佩戴？", # 技术问题 - VR头显
        "我收到的商品破损了，要求退换货。", # 售后问题
        "我的快递少发了一件，物流三天没动静了。", # 物流问题
        "你好，非常感谢你的热情服务！" # 其他/闲聊
    ]

    results_text = "测试结果：\n\n"

    for i, q in enumerate(test_questions, 1):
        print(f"正在测试第 {i} 个问题: {q}")
        res = agent.analyze_and_answer(q)
        results_text += f"【问题 {i}】 {q}\n"
        results_text += f"【回答】 {res['ret']}\n"
        if res.get('image_list'):
            results_text += f"【图片】 {', '.join(res['image_list'])}\n"
        results_text += "-"*40 + "\n"

    with open("test_results.md", "w", encoding="utf-8") as f:
        f.write(results_text)

    print("\n测试完成，结果已保存到 test_results.md")
