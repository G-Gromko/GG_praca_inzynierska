import glob
import random
from sys import platform, argv
import os


def main():
  args = argv[1:]
  
  if len(args) == 0:
      print("Brakuje sciezki do rozpakowanego zbioru danych")
      return
  elif len(args) > 1:
      print("Podano zbyt wiele argumentow")
      return
  elif os.path.isfile(args[0]):
      print("Podano sciezke do pliku zamiast katalogu ze zbiorem danych")
      return
  elif not os.path.exists(args[0]):
      print("Podany katolog nie istnieje")
      return

  dir_replace = args[0]
  print(dir_replace)

  if platform == "linux":
    dir_find = dir_replace + "/**/*.*"
  elif platform == "win32":
    dir_find = dir_replace + "\**\*.*"

  print(dir_find)

  res = []
  trainval_list = []
  res_bekern = []

  for path in glob.glob(dir_find, recursive=True):
      res.append(path)

  for r in res:
      if r.endswith(".bekrn"):
          res_bekern.append(r)

  if platform == "linux":
      for r in res_bekern:
          aux = r.replace(dir_replace, "/content/Data/GrandstaffDataset")
          trainval_list.append(aux)
  elif platform == "win32":
      for r in res_bekern:
          aux = r.replace(dir_replace, "/content/Data/GrandstaffDataset")
          aux2 = aux.replace("\\", "/")
          trainval_list.append(aux2)

  list_test = []

  while len(list_test) != 7661:
      idx = random.randint(0, len(trainval_list)-1)
      aux = trainval_list[idx]
      list_test.append(aux)
      trainval_list.remove(aux)

  list_test.sort()

  paritions_dir = os.path.join(dir_replace, "partitions")
  os.makedirs(paritions_dir, exist_ok=True)

  train_path = os.path.join(paritions_dir, "train.txt")
  val_path = os.path.join(paritions_dir, "val.txt")
  test_path = os.path.join(paritions_dir, "test.txt")

  with open (train_path, 'w') as train:
      for r in trainval_list:
          if r == trainval_list[-1]:
              train.write(f"{r}")
          else:
              train.write(f"{r}\n")
  train.close()

  with open (val_path, 'w') as val:
      for r in trainval_list:
          if r == trainval_list[-1]:
              val.write(f"{r}")
          else:
              val.write(f"{r}\n")
  val.close()

  with open (test_path, 'w') as test:
      for r in list_test:
          if r == list_test[-1]:
              test.write(f"{r}")
          else:
              test.write(f"{r}\n")
  test.close()

if __name__ == "__main__":
    main()