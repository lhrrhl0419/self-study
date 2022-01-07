//luogu P4779 【模板】单源最短路径（标准版）
#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include<queue>
#define MAXN 100022
#define MAXM 400022
#define INF (1<<30)
using namespace std;
struct e { int next, to, dis; } edge[MAXM];
struct node {
	int p, dis;
};
struct cmp {
	bool operator()(node a, node b) {
		return a.dis > b.dis;
	}
};
priority_queue<node, vector<node>, cmp>pq;
int head[MAXN], dis[MAXN];
bool vis[MAXN];
int edgenum = 0;
int n, m, s, u, v, w;
void addedge(int f, int t, int dis)
{
	edge[++edgenum] = { head[f],t,dis };
	head[f] = edgenum;
	return;
}
void dijkstra()
{
	pq.push({ s,0 });
	dis[s] = 0;
	int p, t;
	while (pq.empty() == false)
	{
		p = pq.top().p;
		pq.pop();
		if (vis[p])
			continue;
		vis[p] = true;
		for (int i = head[p]; i; i = edge[i].next)
		{
			t = edge[i].to;
			if (vis[t])continue;
			if (dis[t] > dis[p] + edge[i].dis)
				dis[t] = dis[p] + edge[i].dis, pq.push({ t,dis[t] });
		}
	}
	return;
}
int main()
{
	scanf("%d%d%d", &n, &m, &s);
	for (int i = 1; i <= n; i++)dis[i] = INF;
	for (int i = 1; i <= m; i++)scanf("%d%d%d", &u, &v, &w), addedge(u, v, w);
	dijkstra();
	for (int i = 1; i <= n; i++)printf("%d ", dis[i]);
	return 0;
}
