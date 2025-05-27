"""RAG agent module to select the mode of the RAG agent."""

from collections.abc import Callable


class RagAgentModeEnum:
    """Enum for the mode of the RAG agent."""

    VectorStore = "vectorstore"
    KG = "kg"
    API_BLAST = "api_blast"
    API_ONCOKB = "api_oncokb"
    API_SCANPI = "api_scanpy"


class RagAgent:
    def __init__(
        self,
        mode: str,
        model_name: str | None = "gpt-3.5-turbo",
        connection_args: dict | None = None,
        n_results: int | None = 3,
        use_prompt: bool | None = False,
        schema_config_or_info_dict: dict | None = None,
        conversation_factory: Callable | None = None,
        embedding_func: object | None = None,
        documentids_workspace: list[str] | None = None,
        agent_desc: str | None = None,
        use_reflexion: bool | None = False,
    ) -> None:
        ######
        ##TO DO
        ######
        # mode: 'api' for the case where the agent is querying an API
        # use_prompt: default TRUE for self.mode == api
        """Create a RAG agent that can return results from a database or vector
        store using a query engine.

        Args:
        ----
            mode (str): The mode of the agent. Either "kg" or "vectorstore".

            model_name (str): The name of the model to use.

            connection_args (dict): A dictionary of arguments to connect to the
                database or vector store. Contains database name (in case of
                multiple DBs in one DBMS), host, port, user, and password.

            n_results: the number of results to return for method
                generate_response

            use_prompt (bool): Whether to use the prompt for the query. If
                False, will not retrieve any results and return an empty list.

            schema_config_or_info_dict (dict): A dictionary of schema
                information for the database. Required if mode is "kg".

            conversation_factory (Callable): A function used to create a
                conversation for creating the KG query. Required if mode is
                "kg".

            embedding_func (object): An embedding function. Required if mode is
                "vectorstore".

            documentids_workspace (Optional[List[str]], optional): a list of
                document IDs that defines the scope within which similarity
                search occurs. Defaults to None, which means the operations will
                be performed across all documents in the database.

            agent_desc (str): A description of the agent. If not provided, the
                agent will generate a description based on the mode.

            use_reflexion (bool): Whether to use the ReflexionAgent to generate
                the query.

        """
        self.mode = mode
        self.model_name = model_name
        self.use_prompt = use_prompt
        self.n_results = n_results
        self.documentids_workspace = documentids_workspace
        self.last_response = []
        self._agent_desc = agent_desc

        if self.mode == RagAgentModeEnum.API_SCANPI:
            from .api_agent.base.api_agent import APIAgent
            from .api_agent.python.scanpy.agent import (
                ScanpyFetcher,
                ScanpyInterpreter,
                ScanpyQueryBuilder,
            )
            conversation = conversation_factory()
            self.agent = APIAgent(
                query_builder=ScanpyQueryBuilder(conversation),
                fetcher=ScanpyFetcher(conversation),
                interpreter=ScanpyInterpreter(conversation),
            )
            self.query_func = self.agent.execute        
        else:
            raise ValueError(
                "Invalid mode. Currently, we only support 'api_scanpy'.",
            )

    @property
    def agent_description(self):
        return self._agent_desc

    @agent_description.setter
    def agent_description(self, val: str | None = None):
        self._agent_desc = val

    def generate_responses(self, user_question: str) -> list[tuple]:
        """Run the query function according to the mode and return the results in a
        uniform format (list of tuples, where the first element is the text for
        RAG and the second element is the metadata).

        Args:
        ----
            user_question (str): The user question.

        Returns:
        -------
            results (List[tuple]): A list of tuples containing the results.

        Todo:
        ----
            Which metadata are returned?

        """
        self.last_response = []
        if not self.use_prompt:
            return []
        if self.mode in [
            RagAgentModeEnum.API_SCANPI,
        ]:
            final_answer = self.query_func(user_question)
            if final_answer is not None:
                response = [(final_answer, "response")]
            else:
                response = [(final_answer, "error")]

        else:
            raise ValueError(
                "Invalid mode. Currently, we only support 'api_scanpy'.",
            )
        self.last_response = response
        return response

    def get_description(self):
        if self.agent_description is not None:
            return self.agent_description
        if self.mode == RagAgentModeEnum.KG:
            return self.agent.get_description()
        elif self.mode == RagAgentModeEnum.VectorStore:
            return self.agent.get_description(self.documentids_workspace)
        elif self.mode == RagAgentModeEnum.API_BLAST:
            tool_name = "BLAST"
            tool_desc = (
                "The Basic Local Alignment Search Tool (BLAST) "
                "finds regions of local similarity between sequences. "
                "BLAST compares nucleotide or protein sequences to "
                "sequence databases and calculates the statistical "
                "significance of matches."
            )
            return self.agent.get_description(tool_name, tool_desc)
        elif self.mode == RagAgentModeEnum.API_ONCOKB:
            tool_name = "OncoKB"
            tool_desc = (
                "OncoKB is a precision oncology knowledge base "
                "and contains information about the effects "
                "and treatment implications of specific cancer gene alterations."
            )
            return self.agent.get_description(tool_name, tool_desc)
        else:
            raise ValueError(
                "Invalid mode. Choose either 'kg', 'vectorstore', 'api_blast', or 'api_oncokb'.",
            )
