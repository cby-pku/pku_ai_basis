////自主实现mcts-hex搜索
//#include<iostream>
//#include<cstring>
//#include<algorithm>
//#include<cstdio>
//#include<random>
//#include<climits>
//#include<string>
//#include<ctime>
//#include<math.h>
//#include<algorithm>
//#include<stdlib.h>
//#include "./jsoncpp/json.h"
//#define CHESS 121
//const int N = 11;
//using namespace std;
//inline bool inBoard(int x, int y) {
//	return x >= 0 && y >= 0 && x < N&&y < N;
//}
//
//bool put_available(int Board[N][N], int x, int y) {
//	bool flag = false;
//	if (inBoard(x, y) && Board[x][y] == 0) {
//		flag = true;
//	}
//	return flag;
//}
//
////并查集算法
//class tree;
//class UnionFindSet {
//private:
//	std::vector<int>color;
//public:
//	friend class tree;
//	UnionFindSet(int n) {
//		for (int i = 0; i < n; i++) {
//			color.push_back(i);
//		}
//	}
//	UnionFindSet() = default;
//	int find(int x) {
//		return x == color[x] ? x : (color[x] = find(color[x]));
//	}
//	void join(int x, int y) {
//		int cx = find(x), cy = find(y);
//		if (cx != cy) {
//			color[cx] = cy;
//		}
//	}
//};
////查找范围
//
//const int cx[] = { -1,0,1,0,-1,1 };
//const int cy[] = { 0,-1,0,1,1,-1 };
//bool memo_visit[N][N] = { false };//定义一个查询集
//int lian(int Board[N][N], int x, int y, UnionFindSet detector) {//就假设是红方吧，因为红方先手
//	memo_visit[x][y] = true;
//	int tmp = 0;
//	for (int dir = 0; dir < 6; dir++) {
//		int dx = x + cx[dir]; int dy = y + cy[dir];
//		if (inBoard(dx, dy)) {
//			if (Board[dx][dy] == Board[x][y] && !memo_visit[dx][dy]) {
//				//如果棋子颜色相同，就把他们相连接起来
//				tmp++;
//				detector.join(dx*N + dy, x*N + y);
//				tmp += lian(Board, dx, dy, detector);
//			}
//		}
//	}
//	return tmp;
//}
//
////
//int ValidScore(int Board[N][N], int result[N][N]) {
//	memset(result, 0, CHESS * 4);
//	int ans = 0;
//	for (int i = 0; i < N; i++) {
//		for (int j = 0; j < N; j++) {
//			if (put_available(Board, i, j)) {
//				ans++; result[i][j] = 1;
//			}
//		}
//	}
//	return ans;
//}
////mcts算法，定义一次模拟
//class tree {
//public:
//	tree*parent;
//	tree*child[CHESS];//提前开好最大子节点树
//	int Board[N][N];
//	int countChild;
//	int countChildMax;
//	double UCB;//每个臂的ucb值
//	int n;//该节点的被访问次数
//	int *all_countPtr;//总节点数的指针
//	double value;
//	int childAct[CHESS][2];
//	mutable UnionFindSet detector, enemy_detector;
//	tree(tree*fatherptr, int*all, int parentboard[11][11], int enemy_act[2]) :detector(N*N + 2), enemy_detector(N*N + 2) {
//		for (int i = 0; i < 9; i++) {
//			for (int j = 0; j < 9; j++) {
//				Board[i][j] = -parentboard[j][i];//将棋盘对称
//			}
//		}
//		Board[enemy_act[0]][enemy_act[1]] = -1;//获取对手动作
//		all_countPtr = all; parent = fatherptr;
//		countChild = 0; n = 0; value = 0.0;
//		init();
//		for (int i = 0; i < N; i++) {//都假设是红方
//			detector.join(N*N, i);
//			detector.join(N*N + 1, N*N - 1 - i);
//			enemy_detector.join(N*N, i);
//			enemy_detector.join(N*N + 1, N*N - 1 - i);
//		}
//	}
//public:
//	tree*Policy() {
//		if (countChildMax == 0) { return this; }
//		if (countChild < countChildMax) {
//			tree*newNode = new tree(this, all_countPtr, Board, childAct[countChild]);
//			child[countChild] = newNode;
//			countChild++; return newNode;
//		}
//		for (int i = 0; i < countChild; i++) {
//			child[i]->UCB = child[i]->value / double(child[i]->n) +
//				0.9*sqrt(log(double(*all_countPtr)) / double(child[i]->n));
//		}
//		int bestChild = 0;
//		double bestUCB = child[0]->UCB;
//		for (int i = 1; i < countChild; i++) {
//			if (child[i]->UCB > bestUCB) {
//				bestUCB = child[i]->UCB;
//				bestChild = i;
//			}
//		}
//		return child[bestChild]->Policy();
//	}
//	double Simulation() {
//		int enemy_board[N][N];
//		for (int i = 0; i < N; i++) {
//			for (int j = 0; j < N; j++) {
//				enemy_board[i][j] = -Board[j][i];//将棋盘斜对称
//			}
//		}
//		int x = 0; int y = 0;
//		for (int i = 0; i < N; i++) {
//			for (int j = 0; j < N; j++) {
//				if (Board[i][j]) {//一定要保证是棋子颜色
//					x += lian(Board, i, j, detector);
//				}
//				if (enemy_board[i][j]) {
//					y += lian(Board, i, j, enemy_detector);
//				}
//			}
//		}
//		if (detector.find(N*N) == detector.find(N*N + 1)) {
//			x += 10000;
//		}
//		else if (enemy_detector.find(N*N) == enemy_detector.find(N*N + 1)) {
//			y += 10000;
//		}
//		//初始分数，lian中返回的是连接数,然后要返回胜负局
//		return max(x - y,0);
//	}
//	void Backpropagation(double dvalue) {
//		int player = 0;//初始化下棋顺序，代表敌方棋手
//		tree*node = this;
//		while (node != nullptr) {
//			if (player == 1) {
//				node->value += dvalue;
//				player--;
//			}
//			else {
//				node->value -= dvalue; player++;
//			}
//		}
//		node->n++;
//		node = node->parent;
//	}
//private:
//	void init() {
//		int result[N][N];
//		int validnumber = ValidScore(Board, result);
//		int validpos[CHESS];
//		int flag = 0;
//		int tmp1 = 0;
//		for (int i = 0; i < N; i++) {
//			for (int j = 0; j < N; j++) {
//				if (result[i][j]) {
//					validpos[flag++] = i * N + j;
//				}//实现了连接
//			}
//		}
//		for (int i = 0; i < validnumber; i++) {
//			childAct[i][0] = validpos[i] / N;
//			childAct[i][1] = validpos[i] / N;
//		}
//		countChildMax = validnumber;
//	}
//};
//
//int main() {
//	srand(clock());
//	int start = clock();
//	int end = (int)(1.98*(double)CLOCKS_PER_SEC);
//	int board[N][N] = { 0 };
//	bool forced_flag;
//	int flag = 0;
//	string str;
//	int x, y;
//	//读入JSON
//	getline(cin, str);
//	Json::Reader reader;
//	Json::Value input;
//	int count = 0;//总节点数
//	reader.parse(str, input);
//	//分析自己收到的输入和自己过往的输出，并恢复状态
//	int turnID = input["responses"].size();
//	for (int i = 0; i < turnID; i++)
//	{
//		x = input["requests"][i]["x"].asInt(), y = input["requests"][i]["y"].asInt();
//		if (x != -1) board[x][y] = 1;
//		x = input["responses"][i]["x"].asInt(), y = input["responses"][i]["y"].asInt();
//		if (x != -1) board[x][y] = -1;
//	}
//	x = input["requests"][turnID]["x"].asInt(), y = input["requests"][turnID]["y"].asInt();
//	if (x != -1) { board[x][y] = 1; forced_flag = false; }
//	else if (input["requests"][0].isMember("forced_x")) {
//		forced_flag = true;
//		flag = 1;//证明我是红方
//	}
//	else {
//		forced_flag = false;
//	}
//	Json::Value ret;
//	Json::Value action;
//	if (forced_flag) {
//		action["x"] = 1;
//		action["y"] = 2;
//	}
//	else {//开始uct决策
//		int tmp[N][N];
//		int enemy_act[2] = { x,y };
//		if (!flag) {//如果flag没有，说明是蓝方，只是对角线传入棋盘，不要颠倒，否则对于先手棋判断错误
//			for (int i = 0; i < N; i++) {
//				for (int j = 0; j < N; j++) {
//					tmp[i][j] = board[j][i];
//				}
//			}
//		}
//		else {
//			for (int i = 0; i < N; i++) {
//				for (int j = 0; j < N; j++) {
//					tmp[i][j] = board[i][j];
//				}
//			}
//		}
//		tree root(nullptr, &count, tmp, enemy_act);
//		while (clock() - start < end) {
//			count++;
//			tree*leafnode = root.Policy();
//			double temp = leafnode->Simulation();
//			leafnode->Backpropagation(temp);
//		}
//		int ans[CHESS] = { 0 };
//		int bestChildNum = 0;
//		int max_value = INT_MIN;
//		for (int i = 0; i < root.countChild; i++) {
//			if (max_value < root.child[i]->value) {
//				max_value = root.child[i]->value;
//				memset(ans, 0, sizeof(ans));
//				bestChildNum = 0;
//				ans[bestChildNum++] = i;
//			}
//			else if (max_value == root.child[i]->value) {
//				ans[bestChildNum++] = i;
//			}
//		}
//		int random = rand() % (bestChildNum+1);
//		int*bestAction = root.childAct[ans[random]];
//		// 输出决策JSON
//		action["x"] = bestAction[0];
//		action["y"] = bestAction[1];
//	}
//	ret["response"] = action;
//	Json::FastWriter writer;
//	cout << writer.write(ret) << endl;
//	return 0;
//}