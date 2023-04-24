#include "problem/tictactoe.hpp"
#include"utils/random_variables.hpp"
using namespace std;
class SarsaAgent;
class TicTacToePolicy {
public:
	friend class SarsaAgent;
	int operator() (TicTacToeState state) {

		// 轮到X时，其选择第一个可行动作
		if (state.active_player() == 0) {

			return state.action_space()[0];

		}
		else {/*
			double maxvalue = -1.0; int ans = 0;
			int num = state.action_space().size();//获得长度
			for (int i = 0; i < num; i++) {
				if (SarsaAgent::Qlist[state.action_space()[i] - 1] > maxvalue) {
					maxvalue = SarsaAgent::Qlist[state.action_space()[i] - 1];
					ans = i;
				}
			}//选择值函数的最大值
			// 学习得到值函数表之后，把下面这句话替换成为根据值函数表贪心选择动作
			return ans; */
			return state.action_space()[0];
		}
	}

};
TicTacToePolicy policy;
//class SarsaAgent {
//private:
//
//	double gamma; double lr; double eps;
//	RandomVariables rv;
//
//public:
//	friend class TicTacToePolicy;
//	double Qlist[9];//一共就9种选择
//	SarsaAgent(double _gamma = 0.9, double learning = 0.01, double epsilon = 0.1) :
//		gamma(_gamma), lr(learning), eps(epsilon) {
//		for (int i = 0; i < 9; i++) {
//			Qlist[i] = 0.0;
//		}
//	}
//	friend class TicTacToeState;
//	int sample(TicTacToeState state) {
//		int action = 0;
//		if (rv.uniform_real() < (1 - eps)) {
//			action = policy(state);
//		}
//		else {
//			int num = state.action_space().size();
//			action = state.action_space()[rand()%num];//随机选择
//		}
//		return action;
//	}
//	void learn(TicTacToeState state, int action, int reward, TicTacToeState next_state,
//		int next_action, bool done) {
//		double target_Q = 0.0;
//		double predict_Q = Qlist[action];//获得当前动作的价值
//		if (done)
//			target_Q = 1.0*reward;
//		else {
//			target_Q = reward + gamma * Qlist[next_action];
//		}
//		Qlist[action] = predict_Q + lr * (target_Q - predict_Q);
//	}
//};
//void run_episode(SarsaAgent agent) {//学习过程
//	TicTacToeState learn_state;
//	auto action = agent.sample(learn_state);
//	while (true){
//		double reward = learn_state.rewards()[0];
//		TicTacToeState next_state = learn_state.next(action);
//		auto next_action = agent.sample(next_state);
//		bool done = learn_state.done();
//		agent.learn(learn_state, action, reward, next_state,
//			next_action, done);
//		if (done) {
//			break;
//		}
//	}
//
//}
int main() {




	// TODO: 通过与环境多次交互，学习打败X策略的方法
	//通过大的for循环进行状态价值回传

	// 测试O是否能够打败X的策略
	TicTacToeState state;//重新定义一个环境，相当于reset
	while (not state.done()) {
		cout << state.rewards()[0] << endl;
		auto action = policy(state);//这是个整数啊
		//cout << action << endl;
		state = state.next(action);
		state.show();
	}
	cout << state.rewards()[0] << endl;//输出的是1！
	return 0;
}