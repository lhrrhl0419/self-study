//luogu P3375 【模板】KMP字符串匹配
#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#define MAXN 1100000
using namespace std;
int n, m;
char x[MAXN], y[MAXN];
int kmp[MAXN];
void input_str(char* p, int* len)
{
	int temp = 0; char c;
	while (c = getchar())
	{
		if (c == '\n')break;
		if (c == '\r')continue;
		temp++;
		*(p + temp) = c;
	}
	*len = temp;
	return;
}
int main()
{
	input_str(x, &n);
	input_str(y, &m);
	int temp = 0;
	for (int i = 1; i <= m; i++)
	{
		while (y[i] != y[temp + 1] && temp != 0)temp = kmp[temp];
		if (temp + 1 < i && y[temp + 1] == y[i])temp++;
		kmp[i] = temp;
	}
	temp = 0;
	for (int i = 1; i <= n; i++)
	{
		while (x[i] != y[temp + 1] && temp != 0)temp = kmp[temp];
		if (y[temp + 1] == x[i])temp++;
		if (temp == m)
		{
			printf("%d\n", i - m + 1);
			temp = kmp[temp];
		}
	}
	for (int i = 1; i <= m; i++)printf("%d ", kmp[i]);
	printf("\n");
	return 0;
}
