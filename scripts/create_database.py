import sqlite3

conn: sqlite3.Connection = sqlite3.connect('data/games.db')

#drop all the tables
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
for table in cursor.fetchall() :
    print(table[0])
    cursor.execute("DROP TABLE {}".format(table[0]))

conn.execute('''
CREATE TABLE Player
(IDPlayer INTEGER PRIMARY KEY,
Name CHAR(50) UNIQUE
);
''')

conn.execute('''
CREATE TABLE Card
(IDCard INTEGER PRIMARY KEY,
Bonus CHAR(8) NOT NULL,
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
IDCharacter INTEGER PRIMARY KEY,
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
IDGame INTEGER PRIMARY KEY,
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
IDStateGame INTEGER PRIMARY KEY,
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
DeckLevel1Remaining INT NOT NULL,
DeckLevel2Remaining INT NOT NULL,
DeckLevel3Remaining INT NOT NULL,
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
IDStatePlayer INTEGER PRIMARY KEY,
IDGame INT NOT NULL,
TurnNumber INT NOT NULL,
PlayerNumber INT NOT NULL,
TokensWhite INT NOT NULL,
TokensBlue INT NOT NULL,
TokensGreen INT NOT NULL,
TokensRed INT NOT NULL,
TokensBlack INT NOT NULL,
TokensGold INT NOT NULL,
VictoryPoints INT NOT NULL,
ReductionWhite INT NOT NULL,
ReductionBlue INT NOT NULL,
ReductionGreen INT NOT NULL,
ReductionRed INT NOT NULL,
ReductionBlack INT NOT NULL,
ReservedCard1 INT,
ReservedCard2 INT,
ReservedCard3 INT,
FOREIGN KEY (IDGame) REFERENCES Game(IDGame),
FOREIGN KEY (ReservedCard1) REFERENCES Card(IDCard),
FOREIGN KEY (ReservedCard2) REFERENCES Card(IDCard),
FOREIGN KEY (ReservedCard3) REFERENCES Card(IDCard)
);
''')

conn.execute('''
CREATE TABLE Action
(
IDAction INTEGER PRIMARY KEY,
IDGame INT NOT NULL,
TurnNumber INT NOT NULL,
PlayerNumber INT NOT NULL,
Type INT NOT NULL,
IDCard INT,
TakeWhite INT,
TakeBlue INT,
TakeGreen INT,
TakeRed INT,
TakeBlack INT,
TakeGold INT,
GiveWhite INT,
GiveBlue INT,
GiveGreen INT,
GiveRed INT,
GiveBlack INT,
GiveGold INT,
IDCharacter INT,
FOREIGN KEY (IDGame) REFERENCES Game(IDGame),
FOREIGN KEY (IDCard) REFERENCES Card(IDCard),
FOREIGN KEY (IDCharacter) REFERENCES Character(IDCharacter)
);
''')

conn.commit()

conn.close()

print("database creation was successfull")
