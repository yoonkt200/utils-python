import sort

if __name__ == "__main__":
    print(sort.selection_sort([1, 5, 2, 3, 1, 9, 7, 4]))
    print(sort.bubble_sort([1, 5, 2, 3, 1, 9, 7, 4]))
    print(sort.insertion_sort([1, 5, 2, 3, 1, 9, 7, 4]))

    test_list = [1, 5, 22, 3, 9, 7, 12, 4, 1]
    worker = sort.Quick(test_list)
    worker.quick_sort(0, len(test_list)-1)
    worker.get_arr()