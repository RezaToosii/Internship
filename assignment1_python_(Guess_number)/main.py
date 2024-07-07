import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QTextEdit, QMessageBox
from PyQt5.QtGui import QIcon


class Player:
    def __init__(self):
        self.guesses = []
        self.numbers = []
        self.locations = []

    def add_history(self, guess, number, location):
        self.guesses.append(guess)
        self.numbers.append(number)
        self.locations.append(location)

    def history_display(self):
        if len(self.guesses) == 0:
            return 'No guesses'
        history = ""
        for i in range(len(self.guesses)):
            history += f'G = {self.guesses[i]} | N = {self.numbers[i]} | L = {self.locations[i]}\n'
        return history

    def guess(self, guess_num, final_num):
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

        result_message = f'G = {guess_num} | N = {number} | L = {location}'
        additional_message = ""

        if location == 4:
            return 1
        elif location == 3:
            additional_message = "Soo close!!! Hope you find it in the next round."
        elif number == 4:
            additional_message = "Well done, you found all the numbers, but you didn't put them in the right place."
        elif number == 3:
            additional_message = "You found most of the numbers. Try for last one."

        additional_message += f'\nYou found {number} numbers correctly, but only {location} of them are in the right place.'

        return 0, result_message, additional_message


class Game(QWidget):
    def __init__(self):
        super().__init__()
        self.players = []
        self.final_number = []
        self.turn = 0
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Number Guessing Game')

        self.setStyleSheet("""
            QWidget {
                background-color: #F0F0F0;
                font-family: Arial;
            }
            QLabel {
                color: #333333;
                font-size: 16px;
            }
            QLineEdit {
                border: 2px solid #333333;
                border-radius: 5px;
                padding: 5px;
                font-size: 16px;
                color: #333333;
            }
            QPushButton {
                background-color: #333333;
                color: #FFFFFF;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #555555;
            }
            QTextEdit {
                border: 2px solid #333333;
                border-radius: 5px;
                padding: 5px;
                font-size: 16px;
                color: #333333;
            }
        """)

        self.layout = QVBoxLayout()

        self.intro_label = QLabel('Welcome to this game, lets start the game quickly.')
        self.layout.addWidget(self.intro_label)

        self.num_players_label = QLabel('Enter the number of players:')
        self.layout.addWidget(self.num_players_label)
        self.num_players_input = QLineEdit(self)
        self.layout.addWidget(self.num_players_input)

        self.secret_code_label = QLabel('Please write your 4 character secret code:')
        self.layout.addWidget(self.secret_code_label)
        self.secret_code_input = QLineEdit(self)
        self.secret_code_input.setEchoMode(QLineEdit.Password)
        self.layout.addWidget(self.secret_code_input)

        self.start_button = QPushButton('Start Game', self)
        self.start_button.clicked.connect(self.start_game)
        self.layout.addWidget(self.start_button)

        self.turn_label = QLabel('')
        self.layout.addWidget(self.turn_label)

        self.history_display = QTextEdit(self)
        self.history_display.setReadOnly(True)
        self.layout.addWidget(self.history_display)

        self.guess_label = QLabel('Enter your guess:')
        self.layout.addWidget(self.guess_label)
        self.guess_input = QLineEdit(self)
        self.layout.addWidget(self.guess_input)

        self.guess_button = QPushButton('Submit Guess', self)
        self.guess_button.clicked.connect(self.make_guess)
        self.layout.addWidget(self.guess_button)
        self.setWindowIcon(QIcon('logo.png'))

        self.setLayout(self.layout)
        self.show()

    def start_game(self):
        num_player = int(self.num_players_input.text())
        for i in range(num_player):
            self.players.append(Player())
        final_number = self.secret_code_input.text()
        while True:
            if len(final_number) == 4:
                break
            final_number = input('WRONG!!!\nPlease write again your 4 character secret code (like 0025) : ')
        self.final_number = list(final_number)

        self.turn_label.setText(f'Now it\'s the turn of player number {self.turn % len(self.players) + 1}')
        self.update_history_display()

        self.show_turn_message()

    def update_history_display(self):
        turn_player = self.players[self.turn % len(self.players)]
        self.history_display.setText(turn_player.history_display())

    def show_turn_message(self):
        QMessageBox.information(self, 'Player Turn',
                                f'Now it\'s the turn of player number {self.turn % len(self.players) + 1}')

    def make_guess(self):
        guess_number = self.guess_input.text()
        turn_player = self.players[self.turn % len(self.players)]
        result, result_message, additional_message = turn_player.guess(guess_number, self.final_number)

        if result == 1:
            QMessageBox.information(self, 'Game Over',
                                    f'Player number {self.turn % len(self.players) + 1} wins.\nThanks for playing.')
            self.close()
        else:
            QMessageBox.information(self, 'Result', f'{result_message}\n{additional_message}')

            self.turn += 1
            if self.turn >= len(self.players) * 10:
                QMessageBox.information(self, 'Game Over',
                                        f'No one guessed the correct number\nThe correct number was {"".join(self.final_number)}\nThank you for your time')
                self.close()
            else:
                self.turn_label.setText(f'Now it\'s the turn of player number {self.turn % len(self.players) + 1}')
                self.update_history_display()
                self.show_turn_message()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    game = Game()
    sys.exit(app.exec_())
