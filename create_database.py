import sqlite3

conn = sqlite3.connect('games.db')

#drop all the tables
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
for table in cursor.fetchall() :
    print(table[0])
    cursor.execute("DROP TABLE {}".format(table[0]))

conn.execute('''
CREATE TABLE Player
(IDPlayer INT PRIMARY KEY NOT NULL,
Name CHAR(50)
);
''')

conn.execute('''
CREATE TABLE Card
(IDCard INT PRIMARY KEY NOT NULL,
Bonus CHAR(8),
CostWhite INT NOT NULL,
CostBlue INT NOT NULL,
CostGreen INT NOT NULL,
CostRed INT NOT NULL,
CostBlack INT NOT NULL,
VictoryPoints INT NOT NULL,
Level INT NOT NULL
);
''')

conn.execute('''
CREATE TABLE Character
(
IDCharacter INT PRIMARY KEY NOT NULL,
VictoryPoints INT NOT NULL,
CostWhite INT NOT NULL,
CostBlue INT NOT NULL,
CostGreen INT NOT NULL,
CostRed INT NOT NULL,
CostBlack INT NOT NULL
);
''')

conn.execute('''
CREATE TABLE Game
(
IDGame INT PRIMARY KEY NOT NULL,
NbPlayers INT NOT NULL,
P1 INT NOT NULL,
P2 INT NOT NULL,
P3 INT,
P4 INT,
VictoryPoints INT NOT NULL,
NbTurns INT NOT NULL,
Winner INT NOT NULL,
FOREIGN KEY (P1) REFERENCES Player(IDPlayer),
FOREIGN KEY (P2) REFERENCES Player(IDPlayer),
FOREIGN KEY (P3) REFERENCES Player(IDPlayer),
FOREIGN KEY (P4) REFERENCES Player(IDPlayer)
);
''')

conn.execute('''
CREATE TABLE StateGame
(
IDStateGame INT PRIMARY KEY NOT NULL,
IDGame INT NOT NULL,
TurnNumber INT NOT NULL,
CurrentPlayer INT NOT NULL,
TokensWhite INT NOT NULL,
TokensBlue INT NOT NULL,
TokensGreen INT NOT NULL,
TokensRed INT NOT NULL,
TokensBlack INT NOT NULL,
TokensGold INT NOT NULL,
Card1_1 INT,
Card1_2 INT,
Card1_3 INT,
Card1_4 INT,
Card2_1 INT,
Card2_2 INT,
Card2_3 INT,
Card2_4 INT,
Card3_1 INT,
Card3_2 INT,
Card3_3 INT,
Card3_4 INT,
Character1 INT,
Character2 INT,
Character3 INT,
Character4 INT,
Character5 INT,
FOREIGN KEY (IDGame) REFERENCES Game(IDGame),
FOREIGN KEY (Card1_1) REFERENCES Card(IDCard),
FOREIGN KEY (Card1_2) REFERENCES Card(IDCard),
FOREIGN KEY (Card1_3) REFERENCES Card(IDCard),
FOREIGN KEY (Card1_4) REFERENCES Card(IDCard),
FOREIGN KEY (Card2_1) REFERENCES Card(IDCard),
FOREIGN KEY (Card2_2) REFERENCES Card(IDCard),
FOREIGN KEY (Card2_3) REFERENCES Card(IDCard),
FOREIGN KEY (Card2_4) REFERENCES Card(IDCard),
FOREIGN KEY (Card3_1) REFERENCES Card(IDCard),
FOREIGN KEY (Card3_2) REFERENCES Card(IDCard),
FOREIGN KEY (Card3_3) REFERENCES Card(IDCard),
FOREIGN KEY (Card3_4) REFERENCES Card(IDCard),
FOREIGN KEY (Character1) REFERENCES Character(IDCharacter),
FOREIGN KEY (Character2) REFERENCES Character(IDCharacter),
FOREIGN KEY (Character3) REFERENCES Character(IDCharacter),
FOREIGN KEY (Character4) REFERENCES Character(IDCharacter),
FOREIGN KEY (Character5) REFERENCES Character(IDCharacter)
);
''')

conn.execute('''
CREATE TABLE StatePlayer
(
IDStatePlayer INT PRIMARY KEY NOT NULL,
IDGame INT NOT NULL,
TurnNumber INT NOT NULL,
PlayerNumber INT NOT NULL,
IDPlayer INT NOT NULL,
VictoryPoints INT NOT NULL,
TokensWhite INT NOT NULL,
TokensBlue INT NOT NULL,
TokensGreen INT NOT NULL,
TokensRed INT NOT NULL,
TokensBlack INT NOT NULL,
TokensGold INT NOT NULL,
BonusWhite INT NOT NULL,
BonusBlue INT NOT NULL,
BonusGreen INT NOT NULL,
BonusRed INT NOT NULL,
BonusBlack INT NOT NULL,
ReservedChard1 INT,
ReservedChard2 INT,
ReservedChard3 INT,
Character1 INT,
Character2 INT,
Character3 INT,
Character4 INT,
Character5 INT,
FOREIGN KEY (IDGame) REFERENCES Game(IDGame),
FOREIGN KEY (IDPlayer) REFERENCES Player(IDPlayer),
FOREIGN KEY (ReservedChard1) REFERENCES Card(IDCard),
FOREIGN KEY (ReservedChard2) REFERENCES Card(IDCard),
FOREIGN KEY (ReservedChard3) REFERENCES Card(IDCard),
FOREIGN KEY (Character1) REFERENCES Character(IDCharacter),
FOREIGN KEY (Character2) REFERENCES Character(IDCharacter),
FOREIGN KEY (Character3) REFERENCES Character(IDCharacter),
FOREIGN KEY (Character4) REFERENCES Character(IDCharacter),
FOREIGN KEY (Character5) REFERENCES Character(IDCharacter)
);
''')

conn.commit()

conn.close()

