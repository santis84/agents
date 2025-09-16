# Importações necessárias
import operator
from typing import TypedDict, Annotated, List, Union

# Certifique-se de ter o litellm e langchain instalados:
# pip install langchain langchain_community langchain_openai litellm langgraph

from langchain_community.tools import tool
from langchain_core.agents import AgentExecutor
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
from langchain_community.chat_models import ChatLiteLLM

# -----------------------------------------------------------
# 1. Definição das Ferramentas (Tools)
# -----------------------------------------------------------

@tool
def ler_arquivo(file_path: str) -> str:
    """Lê o conteúdo de um arquivo de texto. Útil para agentes que precisam de informações salvas."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"Erro: Arquivo não encontrado em {file_path}"

@tool
def escrever_arquivo(file_path: str, content: str) -> str:
    """Escreve o conteúdo em um arquivo de texto. Útil para agentes que precisam salvar resultados."""
    try:
        with open(file_path, 'w') as f:
            f.write(content)
        return f"Conteúdo salvo com sucesso em {file_path}"
    except Exception as e:
        return f"Erro ao escrever o arquivo: {e}"

@tool
def ferramenta_analise_dados(dados: str) -> str:
    """Ferramenta especialista em análise de dados (exemplo)."""
    # Lógica da ferramenta...
    return f"Dados analisados e resultado gerado: {dados}"

# -----------------------------------------------------------
# 2. Definição do Estado do Grafo
# -----------------------------------------------------------

# O estado é um dicionário que será passado entre os nós do grafo.
class AgentState(TypedDict):
    # A lista de mensagens que compõem a conversa.
    messages: Annotated[List[BaseMessage], operator.add]
    # O próximo agente a ser chamado, decidido pelo orquestrador.
    next_agent: str

# -----------------------------------------------------------
# 3. Definição das Classes de Agentes
# -----------------------------------------------------------

# Classe base para os agentes, com a configuração do LiteLLM
class BaseAgent:
    def __init__(self, name: str, system_prompt: str, tools: list):
        self.name = name
        self.system_prompt = system_prompt
        # Configurando o LiteLLM
        self.llm = ChatLiteLLM(model="ollama/llama3") # Use o modelo de sua preferência (ex: "gpt-4", "gemini-pro", etc.)
        self.tools = tools
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("placeholder", "{messages}")
        ])
        self.agent_runnable = self.create_runnable()

    def create_runnable(self):
        return (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: x["messages"][-1].tool_calls if hasattr(x["messages"][-1], 'tool_calls') else []
            )
            | self.prompt
            | self.llm.bind_tools(self.tools)
        )

# Agente Orquestrador
class OrchestratorAgent(BaseAgent):
    def __init__(self, system_prompt: str, experts: list):
        super().__init__("Orquestrador", system_prompt, tools=[]) # O orquestrador não usa ferramentas, ele só delega.
        self.experts = experts
        self.agent_runnable = self.create_runnable()

    def create_runnable(self):
        # O orquestrador usa um modelo para decidir o próximo agente
        return self.prompt | self.llm

    def decide_next_agent(self, state: AgentState) -> str:
        """Função de roteamento que o orquestrador usa para decidir o próximo nó."""
        last_message = state['messages'][-1].content
        # Lógica de decisão baseada no conteúdo da última mensagem.
        # Você pode usar a LLM para fazer essa decisão com um prompt mais complexo.
        if "analise de dados" in last_message.lower():
            return "analista_dados"
        elif "pesquisa" in last_message.lower():
            return "pesquisador"
        elif "criação de conteúdo" in last_message.lower():
            return "criador_conteudo"
        else:
            return END # Se não for para nenhum especialista, finaliza o processo.

# Agente Pesquisador
class PesquisadorAgent(BaseAgent):
    def __init__(self):
        system_prompt = "Você é um agente especialista em pesquisa. Sua função é buscar informações e consolidá-las."
        tools = [ler_arquivo] # Exemplo de ferramenta
        super().__init__("Pesquisador", system_prompt, tools)
        self.agent_executor = AgentExecutor(agent=self.agent_runnable, tools=self.tools)

# Agente Analista de Dados
class AnalistaDadosAgent(BaseAgent):
    def __init__(self):
        system_prompt = "Você é um agente especialista em análise de dados. Sua função é processar e extrair insights."
        tools = [ler_arquivo, escrever_arquivo, ferramenta_analise_dados] # Exemplo de ferramentas
        super().__init__("Analista de Dados", system_prompt, tools)
        self.agent_executor = AgentExecutor(agent=self.agent_runnable, tools=self.tools)

# -----------------------------------------------------------
# 4. Construção e Compilação do Grafo
# -----------------------------------------------------------

def build_graph() -> CompiledGraph:
    # Instanciando os agentes
    orchestrator = OrchestratorAgent(
        system_prompt="Você é um orquestrador de um time de IA. Sua única função é receber a solicitação do usuário e encaminhá-la para o agente especialista correto. Se a tarefa não se encaixa em nenhum especialista, finalize o processo. A sua resposta deve ser o nome do agente especializado a ser usado.",
        experts=["pesquisador", "analista_dados"]
    )
    pesquisador = PesquisadorAgent()
    analista_dados = AnalistaDadosAgent()

    # Criando o grafo
    workflow = StateGraph(AgentState)

    # Adicionando os nós (agentes)
    workflow.add_node("orquestrador", lambda state: {"messages": [AIMessage(content=orchestrator.decide_next_agent(state))]})
    workflow.add_node("pesquisador", lambda state: {"messages": [pesquisador.agent_executor.invoke(state)]})
    workflow.add_node("analista_dados", lambda state: {"messages": [analista_dados.agent_executor.invoke(state)]})

    # Definindo o ponto de entrada
    workflow.set_entry_point("orquestrador")

    # Definindo as transições (arestas)
    # A transição do orquestrador é baseada no nome do próximo agente.
    workflow.add_conditional_edges(
        "orquestrador",
        lambda state: state['messages'][-1].content, # Pega o nome do agente decidido pelo orquestrador
        {
            "pesquisador": "pesquisador",
            "analista_dados": "analista_dados",
            END: END
        }
    )
    # As transições dos especialistas voltam para o orquestrador ou finalizam
    workflow.add_edge("pesquisador", "orquestrador")
    workflow.add_edge("analista_dados", "orquestrador")
    
    # Compilando o grafo
    return workflow.compile()

# -----------------------------------------------------------
# 5. Execução do Fluxo
# -----------------------------------------------------------

if __name__ == "__main__":
    app = build_graph()
    
    # Exemplo 1: Fluxo de análise de dados
    print("--- Executando a requisição de análise de dados ---")
    inputs1 = {"messages": [HumanMessage(content="Preciso de uma análise de dados sobre o arquivo 'vendas.txt'.")]}
    for s in app.stream(inputs1):
        print(s)
    
    # Exemplo 2: Fluxo de pesquisa (hipotético)
    print("\n--- Executando a requisição de pesquisa ---")
    inputs2 = {"messages": [HumanMessage(content="Faça uma pesquisa sobre o impacto da IA na medicina.")]}
    for s in app.stream(inputs2):
        print(s)
