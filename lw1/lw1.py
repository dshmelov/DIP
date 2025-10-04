import numpy as np


def main() -> None:
    """
    Задание:
    Средствами numpy сгенерировать и найти одинаковые элементы в двух массивах. Сохранить одинаковые элементы в третий массив, а в исходных массивах их обнулить.
    """

    A = np.random.randint(0, 100, (100,))
    B = np.random.randint(0, 100, (100,))

    C = np.intersect1d(A, B)

    A = np.array([0 if v in C else v for v in A])
    B = np.array([0 if v in C else v for v in B])
    pass


if __name__ == '__main__':
    np.random.seed(seed=42)
    main()
