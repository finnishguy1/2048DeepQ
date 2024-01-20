import itertools
import random


class Game2048:
    def __init__(self, height, width):
        self.height = height
        self.width = width

        self.grid = [[0 for _ in range(self.width)] for _ in range(self.width)]
        self.score = 0

    def move(self, dir):
        if dir == "0":
            moved = self.move_left()
        elif dir == "1":
            moved = self.move_up()
        elif dir == "2":
            moved = self.move_right()
        elif dir == "3":
            moved = self.move_down()
        else:
            print("Invalid direction, please try again: ")
            return

        if self.lost():
            print("Game over!")
            return 1

        if self.empty_cells():
            self.add_block()

    def move_left(self):
        # the others are handled as transposes of this case
        moved = False
        for row in self.grid:
            for j in range(1, len(row)):
                k = j - 1
                while k >= 0 and row[k] == 0:
                    row[k] = row[k + 1]
                    row[k + 1] = 0
                    k -= 1
                if k >= 0 and row[k] == row[k + 1]:
                    row[k] *= 2
                    self.score += row[k]
                    row[k + 1] = 0
                    moved = True
        return moved

    def move_up(self):
        self.transpose_grid()
        moved = self.move_left()
        self.transpose_grid()
        return moved

    def move_right(self):
        self.reverse_rows()
        moved = self.move_left()
        self.reverse_rows()
        return moved

    def move_down(self):
        self.transpose_grid()
        moved = self.move_right()
        self.transpose_grid()
        return moved

    def lost(self):
        lost = not self.empty_cells()
        for i in range(self.height):
            for j in range(self.width):
                for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    if (
                        i + dy < 0
                        or i + dy >= self.height
                        or j + dx < 0
                        or j + dx >= self.width
                    ):
                        continue
                    if self.grid[i][j] == self.grid[i + dy][j + dx]:
                        lost = False
        return lost

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

    def transpose_grid(self):
        for i in range(self.height):
            for j in range(self.width):
                if i > j:
                    self.grid[i][j], self.grid[j][i] = self.grid[j][i], self.grid[i][j]

    def reset(self):
        self.grid = [[0 for _ in range(self.width)] for _ in range(self.width)]
        self.score = 0

    def reverse_rows(self):
        for i in range(self.height):
            self.grid[i] = self.grid[i][::-1]

    def score(self):
        return self.score

    def print_grid(self):
        for row in self.grid:
            print(row)
        print()


def main():
    e = Game2048(4, 4)
    while not e.lost():
        x = input("direction: ")
        e.move(x)
        e.print_grid()
    e.print_grid()


if __name__ == "__main__":
    main()
