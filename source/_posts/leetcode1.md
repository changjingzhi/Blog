---
title: leetcode1-数组/字符串
date: 2024-06-05 20:51:56
tags: leetcode
---
[刷题链接](https://leetcode.cn/studyplan/top-interview-150/)
[](https://programmercarl.com/%E6%95%B0%E7%BB%84%E7%90%86%E8%AE%BA%E5%9F%BA%E7%A1%80.html)


## 合并两个有序数组

[合并两个有序数组](https://leetcode.cn/problems/merge-sorted-array/?envType=study-plan-v2&envId=top-interview-150)

### 解法一

```
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        while(m + n ){
            if (m && n)
                nums1[m+n] = nums1[m-1] > nums2[n-1] ? nums1[m---1] : nums2[n---1];
            else if (!m)
                nums1[m+n] = nums2[n---1];
            else
                return;

        }
    }
};

```
将两个有序数组合并成一个有序数组的功能。