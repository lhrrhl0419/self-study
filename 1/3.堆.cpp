//luogu P3378 【模板】堆 
#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#define MAXN 1000010
using namespace std;
int n, op;
int heap[MAXN],num=0;
void heap_swap_up(int p)
{
	if (p == 1)return;
	if (heap[p >> 1] > heap[p])
	{
		swap(heap[p >> 1], heap[p]);
		heap_swap_up(p >> 1);
	}
	return;
}
void heap_swap_down(int p)
{
	int temp;
	if (((p << 1) <= num && heap[p << 1] < heap[p]) || ((p << 1) + 1 <= num && heap[(p << 1) + 1] < heap[p]))
	{
		if ((p << 1) + 1 <= num && heap[(p << 1) + 1] < heap[p << 1])temp = (p << 1) + 1;
		else temp = (p << 1);
		swap(heap[p], heap[temp]);
		heap_swap_down(temp);
	}
	return;
}
int main()
{
	scanf("%d", &n);
	for (int i = 1; i <= n; i++)
	{
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			scanf("%d", &heap[++num]);
			heap_swap_up(num);
			break;
		case 2:
			printf("%d\n", heap[1]);
			break;
		case 3:
			heap[1] = heap[num];
			num--;
			if (num != 0)heap_swap_down(1);
			break;
		}
	}
	return 0;
}
