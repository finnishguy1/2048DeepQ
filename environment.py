import itertools
import random


class Game2048:

    def __init__(self, height, width):
        self.heigh = height
        self.width = width

        self.grid = [[0 for _ in range(self.width)] for _ in range(self.width)]


        def move_left():
            # the others are handled as transpose of this case
            for row in self.grid():
                for j in range(1, len(row)):
                    for i in range(j - 1, -1, -1):


        def add_block(self):
            # if empty_cells is empty, game lost probably
            y, x = random.choice(self.empty_cells())
            num = 2 if random.random() > 0.1 else 4
            self.grid[y][x] = num


        def empty_cells(self):
            empty_cells = []
            for y, x in itertools.product(range(self.height), range(self.width)):
                if self.grid[y][x] == 0:
                    empty_cells.append((y, x))
            return empty_cells






def main():
    e = Game2048(4, 4)
    print(e.grid)


if __name__ == "__main__":
    main()
