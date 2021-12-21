from random import randint


class App:
    # Your code goes here
    def __init__(self, data: [[id, str]]):
      self.data = data
      self.count = 0
      self.prepared_users = []

    def prepare(self, user_id: int) -> [int, int]:
      self.prepared_users.append(user_id)
      password = [tpl for tpl in self.data if tpl[0] == user_id][0][1]
      if password is None:
        return []
      num1 = randint(0, len(password)-1)
      num2 = randint(0, len(password)-1)
      while num2 == num1:
         num2 = randint(0, len(password-1))
      return num1, num2

    def check(self, user_id: int, char1: str, char2: str) -> bool:
      if user_id not in self.prepared_users:
        self.count += 1
        return False
      password = [tpl for tpl in self.data if tpl[0] == user_id][0][1]
      self.count = 0
      return True

    def getStats(self) -> int:
      return self.count


pass1 = "Passw0rd"
passwords = [
  [1, pass1]
]

app = App(passwords)
print(app.getStats())
i1, i2 = app.prepare(1)
print(app.check(1, pass1[i1], pass1[i2]))
print(app.getStats(), 0)
print(app.check(2, "A", "B"))
print(app.getStats(), 1)