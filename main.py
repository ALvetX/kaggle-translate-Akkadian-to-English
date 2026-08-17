import math
import sys
from typing import List, Optional
import collections

# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
#
#
# def build_linked_list(arr: List[int]) -> Optional[ListNode]:
#     """将 Python 数组 (List) 转换为单链表，返回头节点"""
#     if not arr:
#         return None
#
#     # 1. 创建一个虚拟头节点，它的作用是作为一个“固定的锚点”
#     dummy = ListNode(0)
#     # 2. 用一个指针 curr 来遍历并构建链表
#     curr = dummy
#
#     # 3. 遍历数组，依次把元素挂到链表末尾
#     for val in arr:
#         curr.next = ListNode(val)  # 创建新节点并连接
#         curr = curr.next  # 指针往后移动一步
#
#     # dummy.next 就是真正的链表头节点
#     return dummy.next


# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

import torch

def solve():
    # k, m, n, s = map(int, input().split())
    # query = [float(x) for x in input().split()]
    # samples = []
    # for i in range(m):
    #     row = input().split()
    #     if not row:
    #         continue
    #     features = [float(x) for x in row[:n]]
    #     label = int(float(row[-1]))
    #     sq_dist = 0
    #     for q, f in zip(query, features):
    #         tmp = (q - f) ** 2
    #         sq_dist += tmp
    #     samples.append((sq_dist, i, label))
    #
    # samples.sort(key=lambda x: (x[0], x[1]))
    # top_k = samples[:k]
    # freq = collections.defaultdict(int)
    # first_appearance = collections.defaultdict(int)
    # for rank, (_, _, label) in enumerate(top_k):
    #     freq[label] = freq.get(label, 0) + 1
    #     if label in first_appearance:
    #         first_appearance[label] = rank
    # max_freq = max(freq.values())
    # best_label = None
    # best_rank = float("inf")
    #
    # for label, count in freq.items():
    #     if count == max_freq:
    #         if first_appearance[label] < best_rank:
    #             best_rank = first_appearance[label]
    #             best_label = label

    # print(f"{best_label} {max_freq}")

    print(torch.__version__)


if __name__ == '__main__':
    solve()
