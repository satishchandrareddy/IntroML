# write_csv.py
import csv

def write_csv(filename,list_out):
	'''
	Program to write results from list of lists to csv file
	INPUT:
	filename	= (string) name of output file (should have extension .csv)
	list_out 	= (list) list of lists containing information to be output
	RETURNS
	nothing returned - data written to filename
	'''
	with open(filename, 'wt', newline='') as f:
		csv_writer = csv.writer(f)
		csv_writer.writerows(list_out)

if __name__ == "__main__":
	list_out = [["ID", "Name", "Age"],[1,"Jane Doe",23],[2,"John Smith", 15]]
	write_csv("testfile.csv",list_out)