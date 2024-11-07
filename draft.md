```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-


#prim算法
def update(index,nodepair,G,n):
    #函数实现代码开始
    #请在其中写入你的代码  
    for i in range(n):
        if G[index][i]>0: 
            if nodepair[i][1]>G[index][i]:
                nodepair[i][0]=index
                nodepair[i][1]=G[index][i]
    #函数实现代码结束                                
    return nodepair

def prim(G, n):
    INF      = 100000
    visit    = [-1] * n  #遍历情况记录数组
    nodepair =[[-1,INF] for i in range(n)]  #节点伴随二元数组 第一个表示前一个索引preIndex，第二个表示距离
    #先取第一个节点
    visit[0]=1
    print(0,nodepair[0]) #打印第一个节点
    #更新节点伴随二元数组
    nodepair=update(0,nodepair,G,n)
    #print(nodepair)
    #函数实现代码开始
    #请在其中写入你的代码  
    for _ in range(n-1):  
        min_distance = INF
        min_index = -1
        for i in range(n):
            if visit[i]==-1 and nodepair[i][1]<min_distance:
                min_distance = nodepair[i][1]
                min_index = i
        if min_index == -1:
            break
        visit[min_index]=1
        print(min_index,nodepair[min_index])
        nodepair=update(min_index,nodepair,G,n)
                    
    return nodepair

if __name__=='__main__':
        Glist=input().split('-')
        G=[]
        for item in Glist:
                row=item.split(',')
                g=[int(i) for i in row]
                G.append(g)
        n=len(G)
        prim(G,n)


```