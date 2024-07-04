"""
this file is about my first assignment for Internship where I have to implement a simple game.
author: Reza Toosi
"""


class Player:

    def __init__(self):
        """
            Parameter definition

                Parameters
                ----------
                guesses   = list of player guess
                numbers   = list of number player have find
                locations = list of number player find in correct location
        """
        self.guesses = []
        self.numbers = []
        self.locations = []

    def add_history(self, guess, number, location):
        """
            Add player guess to history
        """

        self.guesses.append(guess)
        self.numbers.append(number)
        self.locations.append(location)

    def history_display(self):
        """
            Display the history of player guesses
        """

        if len(self.guesses) == 0:
            print('No guesses')
            return 0
        for i in range(0, len(self.guesses), 2):
            print(f'G = {self.guesses[i]} | N = {self.numbers[i]} | L = {self.locations[i]}')

    def guess(self, guess_num, final_num):
        """
            Compares the player's guess with the secret code and prints the result

                Parameters
                ----------
                location   = number of numbers found in the correct place
                number     = Number of numbers found
                guess_num  = player guess
                temp       = a copy of secret code

                Output
                ----------
                add_history   = Add player guess & result to player history
                some reaction = Print a few sentences in response to the events of the game
                result        = How many numbers are found and how many are in the correct place

                return
                ----------
                0 if the hidden number is not found
                1 if the hidden number is found

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

        if location == 4:
            return 1
        elif location == 3:
            print('soo close!!!\nHope find it in next round.')
        elif number == 4:
            print('Well done, you found all the numbers\nBut you didnt put them in the right place')
        elif number == 3:
            print('You found most of the numbers\nPut them in the right place')

        print(f'You found {number} numbers correctly, but only {location} of them are in the right place')
        return 0


class Game:
    """
        Parameter definition

            Parameters
            ----------
            players        = list of players
            final_number   = secret code
    """
    def __init__(self):
        self.players = []
        self.final_number = []

    def start(self):
        """
            Takes the number of players and sector code from the player
        """

        num_player = int(input('Enter the number of players : '))
        for i in range(num_player):
            self.players.append(Player())
        final_number = input('Please write your 4 character secret code : ')
        while True:
            if len(final_number) == 4:
                break
            final_number = input('WRONG!!!\nPlease write again your 4 character secret code (like 0025) : ')
        self.final_number = list(final_number)

    def run(self):
        """
            Management of players' turn and limitation of the total number of turns
        """
        turn = 0

        while turn < len(self.players) * 10:
            print('-------------------------------------------')

            print(f'Now its the turn of player number {turn % len(self.players) + 1}')
            turn_player = self.players[turn % len(self.players)]
            turn_player.history_display()
            guess_number = input('Enter your guess : ')

            if turn_player.guess(guess_number, self.final_number) == 1:
                print(f'Player number {turn % len(self.players) + 1} wins.')
                print('Thanks for playing.')
                exit()

            turn += 1
        print(f'No one guessed the correct number\nThe correct number was {self.final_number}\nThank you for your time')


print('Welcome to this game, lets start the game quickly')
game = Game()
game.start()
game.run()
