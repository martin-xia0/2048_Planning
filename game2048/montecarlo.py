from typing import Any, Union
import numpy as np
from game2048.game import Game
from threading import Thread


# 顶层投票箱8个不同判决投票决定移动
def board_to_move(board):
    threads = []
    voter = {}
    for i in range(8):
        t = Thread(target=board_to_move_thread, args=(board, i, voter))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    # 统计票箱，获得投票数最多的移动方向
    result = list(voter.values())
    print(result)
    count = [result.count(i) for i in range(4)]
    print(count)
    direction = count.index(max(count))
    print(direction)
    return direction


# 传当前棋盘，初始化game随机跑n个simulation，每个simulation跑k步算分数
# def board_to_move_thread(board, id, voter):
def board_to_move_thread(board):
    space = list(board.flatten()).count(0)
    max_val = max(board.flatten())
    if space > 5:
        direction = intuition(board)
    else:
        direction = exploration(board)
    return direction


def intuition(board):
    # ------------------------------
    # 正常情况，使用直觉快速思考（近似贪心法）
    score_dict = {0:0,1:0,2:0,3:0}
    game = Game(4, 4096, enable_rewrite_board=True)
    for i in range(4):
        # 每次模拟前置零
        game.board = board
        game.score_move = 0
        game.only_move(i)
        # 无合并，就是菜
        # print(game.board)
        if game.score_move == 0:
            score_dict[i] = -100 + board_score(game.board)
        # 有合并，计算权值
        else:
            score_dict[i] += game.score_move + board_score(game.board)
    print('score_dict {}'.format(score_dict))
    direction = max(score_dict, key=score_dict.get)
    return direction


def exploration(board):
    #-------------------------------
    #  危急情况，使用启发式思考（蒙特卡洛方法）
    #  根据危机严重程度分级，确定模拟次数
    space = list(board.flatten()).count(0)
    n = max(400, (10-space)*100)
    score_dict = {0:[],1:[],2:[],3:[]}
    for _ in range(n):
        # 初始化随机移动机
        rand_game = Game(4, 4096, enable_rewrite_board=True)
        rand_game.board = board
        # 首次随机移动
        first_direction = np.random.randint(0, 4)
        rand_game.move_and_score(first_direction)
        score = board_score(rand_game.board)
        while rand_game.end == 0:
            rand_direction = np.random.randint(0, 4)
            rand_game.move_and_score(rand_direction)
        score_dict[first_direction].append(rand_game.score_move+score)
    # print(score_dict)
    # 根据均分最高选择移动方向
    score_ave = {i: sum(score_dict[i])/len(score_dict[i]) for i in range(4)}
    print(score_ave)
    direction = max(score_ave, key=score_ave.get)
    max_score = score_ave[direction]
    min_direction = min(score_ave, key=score_ave.get)
    min_score = score_ave[min_direction]
    # 情况十分危急，超级搜索
    if max_score < 100 or min_score < 50:
        n = 2000
        if max_score < 80:
            n = 3000
        # 存活系数小于50后，每一步都会举步维艰
        if max_score < 50:
            n = 8000
        score_dict = {0: [], 1: [], 2: [], 3: []}
        for _ in range(n):
            # 初始化随机移动机
            rand_game = Game(4, 4096, enable_rewrite_board=True)
            rand_game.board = board
            # 首次随机移动
            first_direction = np.random.randint(0, 4)
            rand_game.move_and_score(first_direction)
            # score = board_score(rand_game.board)
            while rand_game.end == 0:
                rand_direction = np.random.randint(0, 4)
                rand_game.move_and_score(rand_direction)
            score_dict[first_direction].append(rand_game.score_move)
        # print(score_dict)
        # 根据均分最高选择移动方向
        score_ave = {i: sum(score_dict[i]) / len(score_dict[i]) for i in range(4)}
        print(score_ave)
        direction = max(score_ave, key=score_ave.get)
    return direction
    # voter[id] = direction
    # return voter


# 计算棋盘的评估分
def board_score(board):
    # -----------------------------
    # 对数预处理
    board_dic = {0:0, 2:1, 4:2, 8:3, 16:4, 32:5, 64:6, 128:7, 256:8, 512:9, 1024:10, 2048:11, 4096:12}
    board = np.array([board_dic[i] for i in board.flatten()]).reshape(4,4)
    # -----------------------------
    # 1.空格数
    space_score = list(board.flatten()).count(0)
    # print('space_score {}'.format(space_score))

    # 2.单调性(平滑度)
    monotone_score = 0
    board_rot = np.rot90(board, 1)
    # 共获得4行4列8个评分
    for row in board:
        row = [i for i in row if i != 0]
        if len(row) < 2:
            monotone_row = 0
        elif len(row) == 2:
            monotone_row = abs(row[1]-row[0])
        elif len(row) == 3:
            monotone_row = abs(row[2]-row[1])+abs(row[1]-row[0])
        elif len(row) == 4:
            monotone_row = abs(row[3]-row[2])+abs(row[2]-row[1])+abs(row[1]-row[0])
        monotone_score += monotone_row
        # print('monotone_row {}'.format(monotone_row))
    for col in board_rot:
        col = [i for i in col if i != 0]
        if len(col) < 2:
            monotone_col = 0
        elif len(col) == 2:
            monotone_col = abs(col[1]-col[0])
        elif len(col) == 3:
            monotone_col = abs(col[2]-col[1])+abs(col[1]-col[0])
        elif len(col) == 4:
            monotone_col = abs(col[3]-col[2])+abs(col[2]-col[1])+abs(col[1]-col[0])
        monotone_score += monotone_col
        # print('monotone_col {}'.format(monotone_col))
    monotone_score = monotone_score/8
    # print('monotone_score {}'.format(monotone_score))

    # 3.最大值
    max_score = max(board.flatten())
    # print('max_score {}'.format(max_score))

    # 4.良拐角 如果角点是最大值加分
    corner_score = 0
    corner = [board[0][0], board[0][3], board[3][0], board[3][3]]
    corner_score = corner.count(max(board.flatten()))
    # print('corner_score {}'.format(corner_score))

    # 按比例掺杂,获得总成绩
    score = 0
    score = space_score*10 - monotone_score*2 + max_score + corner_score*6
    # print('board score {}'.format(score))
    return score


if __name__ == '__main__':
    board = np.array([[0., 0., 4., 4.],
                      [0., 0., 0., 2.],
                      [0., 4., 4., 8.],
                      [0., 4., 16., 256.]])
    # board_score(board)
    # direction, voter = board_to_move(board)
    # direction = board_to_move_thread(board)
    # print(direction)
# print(random_score(board))
# print(random_to_die(board))
    # game=Game(4,2048,enable_rewrite_board=True)
    # game.board=board
    # print(game.board)
    # score=game.only_move(3)
    # print(game.board)
    # print(game.score_move)



