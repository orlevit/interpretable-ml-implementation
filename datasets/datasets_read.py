import pandas as pd
def read_and_convert_br(loc):
  br_data = pd.read_csv(loc)
  br_data['ctemp'] = br_data.temp* (39 - (-8)) + (-8)
  br_data['cwindspeed'] = br_data.windspeed* 67
  br_data['chum'] = br_data.hum* 100
  X = br_data.loc[:,map(lambda x :x not in ['cnt','dteday'], list(br_data.columns))]
  y = br_data.loc[:,'cnt']
  return(br_data, X ,y)