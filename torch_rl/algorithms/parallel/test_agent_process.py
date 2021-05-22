from torch_rl.utils.vec_env import CloudpickleWrapper


class TestAgentProcess:
    def __init__(
            self,
            make_env,
            render,
            queue_to_tester,
            queue_to_model, pipe_from_model,
            queue_to_tb_writer,
            deterministic=True
    ):
        self._make_env = CloudpickleWrapper(make_env)
        self._render = render
        self._queue_to_tester = queue_to_tester
        self._queue_to_model = queue_to_model
        self._pipe_from_model = pipe_from_model
        self._queue_to_tb_writer = queue_to_tb_writer
        self._deterministic = deterministic

        self._ep_idx = 0

    @staticmethod
    def _env_action(action):
        if type(action) is tuple:
            env_action = tuple([
                [mode[i].cpu().numpy() for mode in action]
                for i in range(len(action[0]))
            ])
        else:
            env_action = action.cpu().numpy()
        return env_action

    def play_episode(self, env):
        observation, done = env.reset(), False
        if self._render:
            env.render()
        memory = None
        ep_reward, ep_len = 0, 0

        while not done:
            self._queue_to_model.put(('act', 'test_agent', ([observation], memory, self._deterministic)))
            act_result = self._pipe_from_model.recv()
            action = act_result['action'][0]
            memory = act_result['memory']
            env_action = self._env_action(action)

            observation, reward, done, info = env.step(env_action, render=self._render)

            ep_reward += reward
            ep_len += 1

        self._ep_idx += 1
        self._queue_to_tb_writer.put((
            'add_scalars',
            ('agents/train_reward/', {'agent_test': ep_reward}, self._ep_idx)
        ))
        self._queue_to_tb_writer.put((
            'add_scalars',
            ('agents/train_ep_len/', {'agent_test': ep_len}, self._ep_idx)
        ))

    def work(self):
        env = self._make_env.x()
        try:
            while self._queue_to_tester.empty():
                self.play_episode(env)
        except KeyboardInterrupt:
            print('test env worker: got KeyboardInterrupt')
        finally:
            env.close()
