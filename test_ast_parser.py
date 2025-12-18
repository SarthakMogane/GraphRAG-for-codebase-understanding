from src.indexing.ast_parser import ASTParser

parser = ASTParser(repo_path="/data/repositories/SMS-Spam-VotingClassifier-",languages=['python'])
list = parser._find_files("py")
print(list)