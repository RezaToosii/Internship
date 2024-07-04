def guess(guess_num, final_num):
    location = 0
    number = 0
    guess_num = list(guess_num)

    for i in range(4):
        if guess_num[i] == final_num[i]:
            location += 1

    if len(set(guess_num)) < 4:
        for i in range(len(final_num)):
            if guess_num[i] in final_num:
                number += 1
                final_num.pop(final_num.index(guess_num[i]))

    else:
        for i in guess_num:
            if i in final_num:
                number += 1

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


final_number = input('Please write your 4 character number : ')
while True:
    if len(final_number) == 4:
        break
    final_number = input('WRONG!!!\nPlease write again your 4 character number (like 0025) : ')
final_number = list(final_number)

players = int(input('Enter the number of players : '))

turn = 0

while turn <= players * 10:
    print('-------------------------------------------')
    print(f'Now its the turn of player number {turn % players + 1}')

    guess_number = input('Enter your guess : ')

    if guess(guess_number, final_number) == 1:
        print(f'Player number {turn % players + 1} wins.')
        print('Thanks for playing.')
        exit()

    turn += 1

# Convert to class. Add player history
