from game2048.game import Game
from game2048.displays import Display
import logging

def single_run(size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game=game, display=Display(), **kwargs)
    agent.play(verbose=True)
    return game.score


if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 4096
    N_TESTS = 1
    logging.basicConfig(filename='test_planning.log', format='%(asctime)s%(levelname)s%(message)s')
    logger = logging.getLogger('test_planning')
    hdlr = logging.FileHandler('test_planning')
    formatter = logging.Formatter('%(asctime)s%(levelname)s%(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    '''====================
    Use your own agent here.'''
    from game2048.agents import MonteCarloAgent as TestAgent
    '''===================='''
    turn = 0
    while turn < 1:
        turn += 1
        scores = []
        for i in range(N_TESTS):
            score = single_run(GAME_SIZE, SCORE_TO_WIN,
                               AgentClass=TestAgent)
            # score = single_run(GAME_SIZE, SCORE_TO_WIN,
            #                    AgentClass=TestAgent)
            scores.append(score)
            info = "turn: {} score: {}".format(turn, score) 
            logger.info(info)
        info = "turn: {} ave_score: {}".format(turn, sum(scores)/N_TESTS) 
        logger.info(info)
        print(info)
