#include "problem/tictactoe.hpp"
#include"utils/random_variables.hpp"
using namespace std;
double Qlist[10] = { 0 };
class TicTacToePolicy{
public:
	TicTacToePolicy() {};
    int operator() (TicTacToeState state){
        if (state.active_player() == 0){
            return state.action_space()[0];
        
        } else {
            // 学习得到值函数表之后，把下面这句话替换成为根据值函数表贪心选择动作
			if (state.action_space().size() == 0) {
				return 0;//防止next_action为0的情况，不能进一步预测，默认估计第一个动作的值，因为O永远不下，所以无影响
			}
			int num = state.action_space().size();
			int ans = state.action_space()[0];//这里越界了
			double maxval = -1.0;
			
			if (RandomVariables::uniform_real() < (1.0 - 0.1)) {
				for (int i = 0; i < num; i++) {
					if (Qlist[state.action_space()[i]] > maxval) {
						maxval = Qlist[state.action_space()[i]];
						ans = state.action_space()[i];
					}
				}
			}
			else {
				ans = state.action_space()[rand() % num];
			}
			return ans;
        }
    }

};
using namespace std;
int main(){
    TicTacToeState state;
    TicTacToePolicy policy;

	
    // TODO: 通过与环境多次交互，学习打败X策略的方法
	for (int i = 0; i < 9; i++) {
		Qlist[i] = 0.0;
	}
	TicTacToeState learn_state[100];
	cout << "learn_episode begin" << endl;
	for (int i = 0; i < 40; i++) {//迭代次数
		//env.reset()
		cout<<"section:";
		cout << i;
		cout << "begin" << endl;
		double predict_Q = 0.0;
		bool flag = 0;
		double target_Q = 0.0;
		while (true) {
			if (flag)break;
			auto action = policy(learn_state[i]);
			bool done = learn_state[i].done();
			double reward = 0.0-learn_state[i].rewards()[0];//因为X胜是正值
			//learn阶段
			predict_Q = Qlist[action];
			if (done) {
				target_Q = reward;//需要在done的时候返回该价值，还真不能删
				Qlist[action] = predict_Q + 10.0*(target_Q - predict_Q);//调大结束时的learning_rate，增大奖惩
				flag = 1;
			}
			else {
				TicTacToeState next_state = learn_state[i].next(action);//有可能是这里越界了
				//cout << policy(next_state) << endl;//在policy调用next_state时出错了
				auto next_action = policy(next_state);
				target_Q = reward + 0.9*Qlist[next_action];//gamma=0.9
				Qlist[action] = predict_Q + 2.0*(target_Q - predict_Q);//前期reward一直为0，lr为2.0大一点无所谓
			//更新动作
				learn_state[i] = learn_state[i].next(action);//更新learn_state
				
			}
		}
		//-------渲染当前的终止状态-----
		cout << "current final learn_state:" << endl;
		learn_state[i].show();
		//------打印Q表格-------
		cout <<"current Qlist:" << endl;
		for (int k = 0; k < 9; k++) {
			cout << Qlist[k] << " ";
		}
		cout << endl;
		//--------打印Q表格------
		cout << "section:";
		cout << i;
		cout << "end" << endl;
	}
	cout << "test_begin" << endl;
    // 测试O是否能够打败X的策略
    while (not state.done()){
        auto action = policy(state);
		/*
		//---------------输出action_space----------------
		cout << action << endl;//action是从0开始编号的
		for (int i = 0; i < state.action_space().size(); i++) {
			cout << state.action_space()[i] << " ";
		}
		cout << endl;//action_space里存的是纯action,且action_space 在减小
		//----------------------------------------------
		*/
        state = state.next(action);
        state.show();
    }
    return 0;
}
