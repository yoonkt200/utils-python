import time

def selection_sort(arr):
    for i in range(len(arr)):
        min = arr[i]
        for j in range(i+1, len(arr)):
            if arr[j] <= min:
                temp = arr[j]
                arr[j] = min
                min = temp
        arr[i] = min
    return arr


def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(len(arr)-1):
            if arr[j] >= arr[j+1]:
                temp = arr[j]
                arr[j] = arr[j+1]
                arr[j+1] = temp
    return arr


def insertion_sort(arr):
    for i in range(1, len(arr)):
        j = i-1
        key = arr[i]
        while key < arr[j] and j >= 0:
            arr[j+1] = arr[j]
            j = j-1
        arr[j+1] = key
    return arr


class Quick:
    def __init__(self, data):
        self._data = data

    def quick_sort(self, start, end):
        if start < end:
            pivot = self.partition(start, end)
            self.quick_sort(start, pivot - 1)
            self.quick_sort(pivot + 1, end)

    def partition(self, start, end):
        pivot = end
        wall = start
        left = start

        while left < pivot:
            if self._data[left] < self._data[pivot]:
                self._data[wall], self._data[left] = self._data[left], self._data[wall]
                wall = wall + 1

            left = left + 1

        self._data[wall], self._data[pivot] = self._data[pivot], self._data[wall]
        pivot = wall

        return pivot

    def get_arr(self):
        print(self._data)