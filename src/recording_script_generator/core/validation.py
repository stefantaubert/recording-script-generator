class ValidationError(Exception):
  pass

class InvalidContentError(ValidationError):
  def __init__(self, text: str, msg: str, *args: object) -> None:
    super().__init__(*args)
    self.text = text
    self.msg = msg
  
  text: str
  msg: str

