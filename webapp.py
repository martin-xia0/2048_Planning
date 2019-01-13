from flask import Flask, jsonify, request
from game2048.displays import Display


def get_flask_app(game, agent):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return app.send_static_file('board.html')

    @app.route("/board", methods=['GET', 'POST'])
    def get_board():
        direction = -1
        control = "USER"
        if request.method == "POST":
            direction = request.json
            if direction == -1:
                direction = agent.step()
                control = 'AGENT'
            game.move(direction)
        return jsonify({"board": game.board.tolist(),
                        "score": game.score,
                        "end": game.end,
                        "direction": direction,
                        "control": control})

    return app


if __name__ == "__main__":
    GAME_SIZE = 4
    SCORE_TO_WIN = 4096
    APP_PORT = 5009
    APP_HOST = "0.0.0.0"

    from game2048.game import Game
    game = Game(size=GAME_SIZE, score_to_win=SCORE_TO_WIN)

    from game2048.agents import MonteCarloAgent
    print("WARNING: You are now using a MonteCarloAgent.")
    agent = MonteCarloAgent(game=game,display=Display())

    print("Run the webapp at http://<any address for your local host>:%s/" % APP_PORT)
    agent.play(verbose=True)
    app = get_flask_app(game, agent)
    app.run(port=APP_PORT, threaded=False, host=APP_HOST)  # IMPORTANT: `threaded=False` to ensure correct behavior
    
    
