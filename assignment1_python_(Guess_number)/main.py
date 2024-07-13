"""
This file is about my first assignment for an internship where I have to implement a simple game.
Author: Reza Toosi
"""

import sys
import random
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QTextEdit, \
    QMessageBox
from PyQt5.QtGui import QIcon


class Player:
    def __init__(self):
        """
            Initialize a Player instance with empty guess history.
        """
        self.guesses = []
        self.numbers = []
        self.locations = []

    def add_history(self, guess, number, location):
        """
            Add player's guess and the result to the history.

            Parameters
            ----------
            guess : list
                The guessed numbers.
            number : int
                Number of correct numbers in the guess.
            location : int
                Number of correct numbers in the correct place.
        """
        self.guesses.append(guess)
        self.numbers.append(number)
        self.locations.append(location)

    def history_display(self):
        """
            Display the history of player guesses.
        """
        if len(self.guesses) == 0:
            return 'No guesses'
        history = ""
        for i in range(len(self.guesses)):
            history += f'G = {self.guesses[i]} | N = {self.numbers[i]} | L = {self.locations[i]}\n'
        return history

    def guess(self, guess_num, final_num):
        """
            Compares the player's guess with the secret code and prints the result.

            Parameters
            ----------
            guess_num : list
                Player's guessed numbers.
            final_num : list
                The secret code numbers.

            Returns
            -------
            int
                0 if the hidden number is not found, 1 if the hidden number is found.
        """
        location = 0
        number = 0
        guess_num = list(guess_num)
        temp = final_num.copy()
        for i in range(4):
            if guess_num[i] == final_num[i]:
                location += 1

        if len(set(guess_num)) < 4:
            for i in range(len(temp)):
                if guess_num[i] in temp:
                    number += 1
                    temp.pop(temp.index(guess_num[i]))
        else:
            for i in guess_num:
                if i in final_num:
                    number += 1

        self.add_history(guess_num, number, location)

        result_message = f'G = {guess_num} | N = {number} | L = {location} '
        additional_message = ""

        if location == 4:
            return 1, result_message, ""
        elif number == 0:
            return 0, result_message, "No number is found."
        elif location == 3:
            additional_message += "Soo close!!! Hope you find it in the next round."
        elif number == 4:
            additional_message += "Well done, you found all the numbers, but you didn't put them in the right place."
        elif number == 3:
            additional_message += "You found most of the numbers. Try for last one."

        additional_message += f'\n\n{number} number found, but only {location} of them are in the right place.'

        return 0, result_message, additional_message


class Game(QWidget):
    def __init__(self):
        """
            Initialize the Game instance with an empty list of players and final number.
        """
        super().__init__()
        self.players = []
        self.final_number = []
        self.turn = 0
        self.initUI()
        self.show_turn = True

    def random_num(self):
        random_number = random.randint(1, 9999)
        random_number = f"{random_number:04}"
        self.secret_code_input.setText(random_number)

    def reset(self):
        self.final_number = []
        self.turn = 0
        self.show_turn = True
        self.players.clear()
        self.num_players_input.clear()
        self.num_turn_input.clear()
        self.secret_code_input.clear()
        self.turn_label.clear()
        self.history_display.clear()
        self.guess_remain_label.clear()

    def initUI(self):
        """
            Initialize the UI components and layout.
        """
        self.setWindowTitle('Number Guessing Game')

        self.setStyleSheet("""
            QWidget {
                background-color: #F0F0F0;
                font-family: "Times New Roman", Times, serif;
            }
            QLabel {
                color: #333333;
                font-size: 20px;
            }
            QLineEdit {
                border: 2px solid #333333;
                border-radius: 5px;
                padding: 5px;
                font-size: 20px;
                color: #333333;
            }
            QPushButton {
                background-color: #333333;
                color: #FFFFFF;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-size: 20px;
            }
            QPushButton:hover {
                background-color: #555555;
            }
            QTextEdit {
                border: 2px solid #333333;
                border-radius: 5px;
                padding: 5px;
                font-size: 20px;
                color: #333333;
            }
        """)

        self.layout = QVBoxLayout()

        self.intro_label = QLabel('Welcome to this game, lets start the game quickly.')
        self.layout.addWidget(self.intro_label)

        self.layoutHV1 = QVBoxLayout()
        self.num_players_label = QLabel('Number of players:')
        self.num_players_input = QLineEdit(self)
        self.layoutHV1.addWidget(self.num_players_label)
        self.layoutHV1.addWidget(self.num_players_input)

        self.layoutHV2 = QVBoxLayout()
        self.num_turn_label = QLabel('Max turn:')
        self.num_turn_input = QLineEdit(self)
        self.layoutHV2.addWidget(self.num_turn_label)
        self.layoutHV2.addWidget(self.num_turn_input)

        self.layoutVH1 = QHBoxLayout()

        self.layoutVH1.addLayout(self.layoutHV1)
        self.layoutVH1.addLayout(self.layoutHV2)
        self.layout.addLayout(self.layoutVH1)

        self.layoutHV3 = QVBoxLayout()
        self.secret_code_label = QLabel('Secret code:')
        self.secret_code_input = QLineEdit(self)
        self.secret_code_input.setEchoMode(QLineEdit.Password)
        self.layoutHV3.addWidget(self.secret_code_label)
        self.layoutHV3.addWidget(self.secret_code_input)

        self.layoutHV4 = QVBoxLayout()
        self.random_label = QLabel('')
        self.random_button = QPushButton('Random', self)
        self.random_button.clicked.connect(self.random_num)
        self.layoutHV4.addWidget(self.random_label)
        self.layoutHV4.addWidget(self.random_button)

        self.layoutVH2 = QHBoxLayout()
        self.layoutVH2.addLayout(self.layoutHV3)
        self.layoutVH2.addLayout(self.layoutHV4)

        self.layout.addLayout(self.layoutVH2)

        self.start_button = QPushButton('Start Game', self)

        self.start_button.clicked.connect(self.start_game)

        self.layout.addWidget(self.start_button)

        self.turn_label = QLabel('')
        self.guess_remain_label = QLabel('')
        self.layout.addWidget(self.turn_label)
        self.layout.addWidget(self.guess_remain_label)

        self.history_display = QTextEdit(self)
        self.history_display.setReadOnly(True)
        self.layout.addWidget(self.history_display)

        self.guess_label = QLabel('Enter your guess:')
        self.layout.addWidget(self.guess_label)
        self.guess_input = QLineEdit(self)
        self.layout.addWidget(self.guess_input)

        self.layoutHV5 = QHBoxLayout()

        self.guess_button = QPushButton('Submit Guess', self)
        self.guess_button.clicked.connect(self.make_guess)
        self.reset_button = QPushButton('Reset', self)
        self.reset_button.clicked.connect(self.reset)

        self.layoutHV5.addWidget(self.guess_button)
        self.layoutHV5.addWidget(self.reset_button)

        self.layout.addLayout(self.layoutHV5)

        self.setWindowIcon(QIcon('logo.png'))

        self.setLayout(self.layout)

        self.setGeometry(500, 150, 500, 800)

        self.show()

    def start_game(self):
        """
            Start the game by initializing players and the secret code.
        """

        if self.num_players_input.text() == '' or self.num_turn_input.text() == '' or self.secret_code_label.text() == '':
            QMessageBox.information(self, 'WRONG!!!',
                                    'Fill all the required field.')
            return 0
        num_player = int(self.num_players_input.text())
        if num_player == 1:
            self.show_turn = False
        for i in range(num_player):
            self.players.append(Player())

        final_number = self.secret_code_input.text()
        if len(final_number) != 4:
            QMessageBox.information(self, 'WRONG!!!',
                                    'Please write again your 4 character secret code (like 0025).')
            return 0

        self.final_number = list(final_number)

        self.turn_label.setText(f'Now it\'s the turn of player number {self.turn % len(self.players) + 1}')
        self.guess_remain_label.setText(f'{int(self.num_turn_input.text()) - int(self.turn / len(self.players))}'
                                        f' guesses remain.')
        self.update_history_display()

        if self.show_turn:
            self.show_turn_message()

    def update_history_display(self):
        turn_player = self.players[self.turn % len(self.players)]
        self.history_display.setText(turn_player.history_display())

    def show_turn_message(self):
        QMessageBox.information(self, 'Player Turn',
                                f'Now it\'s the turn of player number {self.turn % len(self.players) + 1}')

    def make_guess(self):
        guess_number = self.guess_input.text()
        if len(self.players) == 0:
            QMessageBox.information(self, 'WRONG!!!', 'Please press the start game button')
            self.guess_input.clear()
            return 0
        if len(guess_number) != 4:
            QMessageBox.information(self, 'WRONG!!!', 'Please write again your guess must be 4 character (like 0025).')
            return 0
        turn_player = self.players[self.turn % len(self.players)]
        result, result_message, additional_message = turn_player.guess(guess_number, self.final_number)
        self.guess_input.clear()
        if result == 1:
            additional_message = f'Player number {self.turn % len(self.players) + 1} won.\nThanks for playing.'
            QMessageBox.information(self, 'Game Over', f'{result_message}\n{additional_message}')
            self.reset()
        else:
            QMessageBox.information(self, 'Result', f'{result_message}\n{additional_message}')

            self.turn += 1
            if self.turn >= len(self.players) * int(self.num_turn_input.text()):
                QMessageBox.information(self, 'Game Over',
                                        f'No one guessed the correct number\nThe correct number was '
                                        f'{"".join(self.final_number)}\nThank you for your time')
                self.reset()
            else:

                self.turn_label.setText(f'Now it\'s the turn of player number {self.turn % len(self.players) + 1}')
                self.guess_remain_label.setText(f'{int(self.num_turn_input.text())- int(self.turn/len(self.players))}'
                                                f' guesses remain.')
                self.update_history_display()
                if self.show_turn:
                    self.show_turn_message()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    game = Game()
    sys.exit(app.exec_())
