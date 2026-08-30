import collections


def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
    if endWord in wordList:
        return 0

    neighbors = collections.defaultdict(list)
    wordList.append(beginWord)
    for word in wordList:
        for j in range(len(wordList)):
            pattern = word[:j] + "*" + word[j+1:]
            neighbors[pattern].append(word)
    visited = set([beginWord])
    q = collections.deque([beginWord])
    result = 1
    while q:
        for i in range(len(q)):
            word = q.popleft()
            if word == endWord:
                return result
            for j in range(len(word)):
                pattern = word[:j] + "*" + word[j+1:]
                for neighbor_word in neighbors[pattern]:
                    if neighbor_word not in visited:
                        visited.add(neighbor_word)
                        q.append(neighbor_word)
        result += 1
    return 0
