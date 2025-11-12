# Chore: Add Precise Type Annotations to All Python Files

## Chore Description
Add comprehensive and precise type annotations to all Python files in the Splendor IA project. This includes:
- Using specific generic types like `List[Card]`, `Dict[int, int]`, `Tuple[int, ...]` instead of generic `list`, `dict`, `tuple`
- Adding type hints to all function parameters and return values
- Using `Optional[Type]` for nullable values
- Using `Union[Type1, Type2]` for values that can be multiple types
- Importing required types from `typing` module (`List`, `Dict`, `Tuple`, `Optional`, `Union`, `Callable`, `Any`)
- Ensuring consistency across all files in the project

The goal is to improve code clarity, enable better IDE support, and catch potential type-related bugs early.

## Relevant Files
Use these files to resolve the chore:

- **board.py** - Main game state class with complex data structures
  - Contains lists of cards, players, and tokens
  - Methods returning various collection types
  - Needs types for: decks (List[List[Card]]), tokens (List[int]), displayedCards (List[List[Card]]), players (List[Player]), characters (List[Character])

- **player.py** - Player class with inventory and state management
  - Contains token lists, card lists, character lists
  - Methods with return types for victory points, bonuses, costs
  - Needs types for: tokens (List[int]), built (List[Card]), reserved (List[Card]), characters (List[Character]), IA (Optional[str])

- **cards.py** - Card class definition
  - Simple class but needs precise types for cost (List[int])
  - Methods returning strings for display

- **characters.py** - Character (nobles) class definition
  - Similar to Card, needs types for cost (List[int])
  - Methods returning strings for display

- **move.py** - Move action class
  - Needs types for action (Union[List[int], Card, int]), tokensToRemove (List[int]), character (Optional[Character])

- **node.py** - MCTS tree node class
  - Needs types for childNodes (List[Node]), move (Optional[Move]), parentNode (Optional[Node]), playerJustMoved (Optional[Player])

- **ISMCTS.py** - Monte Carlo Tree Search implementation
  - Function signatures need precise types
  - Return types for ISMCTS functions (Move and Node respectively)

- **main.py** - Main game loop and database operations
  - Database functions need precise types for parameters and returns
  - History tracking needs List types
  - Functions like getColNames, getPlayersID, loadCards, loadCharacters need return type annotations

- **constants.py** - Constants and utility functions
  - Type aliases for common types could be defined here
  - Functions like strColor, fullStrColor, getColor need return type annotations (str)

- **custom_operators.py** - List utility functions
  - Functions need precise signatures: add(l1: List[int], l2: List[int]) -> List[int]
  - flatten function needs generic typing

- **create_database.py** - Database schema creation script
  - Minimal typing needed (mostly sqlite3 operations)
  - Connection and cursor types can be added

- **load_database.py** - Database initialization script
  - Minimal typing needed (mostly sqlite3 operations)
  - Insert functions can have parameter types

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### Step 1: Add typing imports and type aliases to constants.py
- Import necessary types from typing module at the top of the file
- Add return type annotations to all functions: `strColor(color: int) -> str`, `fullStrColor(color: int) -> str`, `getColor(color: int) -> str`
- Consider adding type aliases for common types like `TokenList = List[int]`, `CostList = List[int]` (for use in other files if needed)

### Step 2: Add precise typing to custom_operators.py
- Import `List`, `TypeVar`, `Any` from typing module
- Add type annotations: `add(l1: List[int], l2: List[int]) -> List[int]`
- Add type annotations: `substract(l1: List[int], l2: List[int]) -> List[int]`
- Add generic type annotations: `flatten(l: List[List[Any]]) -> List[Any]` or use TypeVar for more precision

### Step 3: Add precise typing to cards.py
- Import `List` from typing module
- Add type annotations to `__init__`: `__init__(self, victoryPoints: int, bonus: int, cost: List[int], lvl: int) -> None`
- Add return type annotations to all methods: `isVisible() -> bool`, `setVisible() -> None`, `show() -> None`, `getShow() -> str`
- Add precise types for instance variables in docstrings or as class-level annotations

### Step 4: Add precise typing to characters.py
- Import `List` from typing module
- Add type annotations to `__init__`: `__init__(self, victoryPoints: int, cost: List[int]) -> None`
- Add return type annotations: `__str__() -> str`, `__repr__() -> str`, `getShow() -> str`
- Add equality method types: `__eq__(self, other: object) -> bool`, `__ne__(self, other: object) -> bool`

### Step 5: Add precise typing to move.py
- Import `List`, `Optional`, `Union` from typing module
- Import forward references for Card and Character (use `TYPE_CHECKING` to avoid circular imports)
- Add type annotations to `__init__`: `__init__(self, actionType: int, action: Union[List[int], Card, int], tokensToRemove: List[int], characterSelected: Optional[Character]) -> None`
- Add return type annotations: `__repr__() -> str`, `__eq__(self, other: object) -> bool`, `__ne__(self, other: object) -> bool`

### Step 6: Add precise typing to node.py
- Import `List`, `Optional` from typing module
- Import forward references for Move and Player (use `TYPE_CHECKING` to avoid circular imports)
- Add type annotations to `__init__`: `__init__(self, move: Optional[Move] = None, parent: Optional[Node] = None, playerJustMoved: Optional[Player] = None) -> None`
- Add return types: `getUntriedMoves(self, legalMoves: List[Move]) -> List[Move]`
- Add return types: `UCBSelectChild(self, legalMoves: List[Move], exploration: float = 0.7) -> Node`
- Add return types: `addChild(self, m: Move, p: Player) -> Node`
- Add return types: `update(self, terminalState: Board) -> None` (Board forward reference)
- Add return types: `__repr__() -> str`, `treeToString(self, indent: int) -> str`, `indentString(self, indent: int) -> str`, `childrenToString() -> str`

### Step 7: Add precise typing to player.py
- Import `List`, `Optional`, `Any` from typing module
- Import forward references using `TYPE_CHECKING`
- Add type annotations to `__init__`: `__init__(self, name: str, IA: Optional[str]) -> None`
- Add instance variable type hints using class-level annotations or in `__init__`
- Add return types for all methods: `isHuman() -> bool`, `haveUnseenCards() -> bool`, `getUnseenCards() -> List[Card]`, `removeUnseenCards() -> None`
- Add return types: `getShowBonus() -> List[str]`, `getShowTokens() -> List[str]`, `getShow() -> str`, `show() -> None`, `showReserved() -> str`
- Add return types: `getVictoryPoints() -> int`, `getTotalBonus() -> List[int]`, `realCost(self, card: Card) -> List[int]`, `convertToGold(self, cost: List[int]) -> List[int]`
- Add return types: `realGoldCost(self, card: Card) -> List[int]`, `canBuild(self, card: Card) -> bool`, `gotToManyTokens() -> bool`
- Add return types: `getAllVisible(self, board: Board) -> List[List[Card]]`, `canReserve() -> bool`
- Add parameter and return types for ask* methods that interact with Board

### Step 8: Add precise typing to board.py
- Import `List`, `Optional`, `Union` from typing module
- Import forward references using `TYPE_CHECKING`
- Add type annotations to `__init__`: `__init__(self, nbPlayer: int, IA: List[Optional[str]], debug: bool = False) -> None`
- Add instance variable type hints: decks, tokens, characters, displayedCards, players
- Add return types: `getNbMaxTokens(self, nbPlayer: int) -> int`, `clone() -> Board`, `cloneAndRandomize(self, observer: Player) -> Board`
- Add return types: `getCurrentPlayer() -> Player`, `getNextPlayer(self, currentPlayer: int) -> Player`, `nextPlayer() -> None`
- Add return types: `doMove(self, move: Move) -> None`, `getMoves() -> List[Move]`
- Add return types: `getPossibleTokens() -> List[List[int]]`, `makeMovesTokens(self, allTokens: List[List[int]]) -> List[Move]`
- Add return types: `getPossibleBuild() -> List[Card]`, `makeMovesBuild(self, allBuild: List[Card]) -> List[Move]`
- Add return types: `getPossibleReserve() -> List[Union[Card, int]]`, `makeMovesReserve(self, allReserve: List[Union[Card, int]]) -> List[Move]`
- Add return types: `getVictorious(self, verbose: bool = False) -> int`, `getResult(self, player: Player) -> int`
- Add return types: `checkEndGame(self, verbose: bool = False) -> None`, `getCard(self, move: Move) -> Card`
- Add return types for all action methods: `build()`, `reserve()`, `takeTokens()`, `removeTooManyTokens()`, `takeCharacter()`, `removeCard()`
- Add return types: `show() -> None`, `getShowTokens() -> List[str]`, `getState() -> List[Any]`, `getPlayerState(self, playerNumber: int) -> List[int]`

### Step 9: Add precise typing to ISMCTS.py
- Import `List`, `Optional`, `Tuple` from typing module
- Import forward references for Board, Node, Move
- Add type annotations: `ISMCTS(rootstate: Board, itermax: int, verbose: bool = False, returnTree: bool = False) -> Union[Node, Move]`
- Add type annotations: `ISMCTS_para(rootstate: Board, itermax: int, verbose: bool = False) -> Move`
- Add type annotations: `mergeTrees(originTree: Node, addTree: Node) -> None`

### Step 10: Add precise typing to main.py
- Import `List`, `Tuple`, `Dict`, `Optional`, `Any` from typing module
- Import sqlite3 types (Cursor, Connection)
- Add type annotations: `getColNames(cursor: sqlite3.Cursor, table: str) -> List[str]`
- Add type annotations: `getPlayersID(conn: sqlite3.Connection, cursor: sqlite3.Cursor, nbIte: List[int], Players: List[str]) -> List[int]`
- Add type annotations: `createGame(cursor: sqlite3.Cursor, playersID: List[int], state: Board, winner: int) -> int`
- Add type annotations: `loadCards(cursor: sqlite3.Cursor) -> Dict[Tuple[int, int, int, int, int], int]`
- Add type annotations: `loadCharacters(cursor: sqlite3.Cursor) -> Dict[Tuple[int, int, int, int, int], int]`
- Add type annotations: `saveGamesState(cursor: sqlite3.Cursor, history: List[Any], cards: Dict[Tuple[int, int, int, int, int], int], characters: Dict[Tuple[int, int, int, int, int], int], gameID: int) -> None`
- Add type annotations: `savePlayerState(cursor: sqlite3.Cursor, gameID: int, playerPos: int, history: List[List[int]]) -> None`
- Add type annotations: `savePlayerActions(cursor: sqlite3.Cursor, gameID: int, playerPos: int, history: List[Move], cards: Dict[Tuple[int, int, int, int, int], int], characters: Dict[Tuple[int, int, int, int, int], int]) -> None`
- Add type annotations: `saveIntoBdd(state: Board, winner: int, historyState: List[Any], historyPlayers: List[List[List[int]]], historyActionPlayers: List[List[Move]], nbIte: List[int], Players: List[str]) -> None`
- Add type annotations: `PlayGame(nbIte: List[int], Players: List[str]) -> None`

### Step 11: Add minimal typing to database scripts
- **create_database.py**: Add type for conn (sqlite3.Connection)
- **load_database.py**: Add type annotations to functions: `insertCharacter(character: Tuple[int, int, int, int, int, int]) -> None`, `insertAllCharacters() -> None`, `insertCard(card: Tuple[str, int, int, int, int, int, int, int]) -> None`, `insertAllCards() -> None`

### Step 12: Run validation commands
- Run Python syntax check on all modified files to ensure no syntax errors
- Run the game if there's a way to test it (check main.py execution)
- Verify imports are working correctly

## Validation Commands
Execute every command to validate the chore is complete with zero regressions.

- `python3 -m py_compile *.py` - Compile all Python files to check for syntax errors
- `python3 -c "import board; import player; import cards; import characters; import move; import node; import ISMCTS; import main; print('All imports successful')"` - Verify all modules can be imported
- `python3 -m mypy --strict *.py || echo "Note: mypy strict mode may show issues - manual review needed"` - Optional: Run mypy if available (not required for completion)

## Notes
- Use `from typing import TYPE_CHECKING` and conditional imports to avoid circular import issues between modules (e.g., Board -> Player -> Board)
- The typing module is part of Python's standard library since Python 3.5
- For Python 3.9+, you can use built-in generics like `list[int]` instead of `List[int]`, but use `List[int]` for better compatibility
- Some complex types in main.py (like history structures) may need `List[Any]` if the exact structure is too complex
- The `object` type should be used for `__eq__` and `__ne__` methods to match the object protocol
- Consider adding `-> None` explicitly to methods that don't return values for clarity
- The sqlite3.Cursor and sqlite3.Connection types are available from the sqlite3 module
