from queue import Queue


class InterAgentComm:

    def __init__(self):
        self._mailbox = {}
        self._policies = {}

    def register(self, agent_id, policy):
        self._policies[agent_id] = policy
        self._mailbox[agent_id] = Queue()

    def broadcast(self, sending_agent, msg):
        for agent in self._mailbox:
            if agent != sending_agent:
                self._mailbox[agent].put(msg)
                self.post_message(agent, msg)

    def post_message(self, agent, msg):
        self._mailbox[agent].put(msg)

    def get_message(self, agent):
        try:
            return self._mailbox[agent].get_nowait()
        except:
            return None
