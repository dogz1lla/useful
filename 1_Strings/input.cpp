#include <cmath>
#include <iostream>
#include <cstring>
#include <string>
#include <vector>

using namespace std;

void inspect_input(std::string s) {
  std::string mys = s;
  int l = mys.length();
  cout << "The number of characters in the input line is " << l << "." << endl;

  std::vector<int> character_types(6);
  for(int i=0; i<l; i++){
    int code = int(mys[i]);
    if(code >= 48 && code <= 57)
      character_types[0]++;
    else {
      if(code >= 65 && code <= 90)
	character_types[1]++;
      else {
	if(code >= 97 && code <= 122)
	  character_types[2]++;
	else {
	  if((code >= 33 && code <= 47) ||
	     (code >= 58 && code <= 64) ||
	     (code >= 91 && code <= 96) ||
	     (code >= 123 && code <= 126))
	    character_types[3]++;
	  else {
	    if(code == 32)
	      character_types[4]++;
	    else
	      character_types[5]++;
	  }
	}
      }
    }
  }

  cout << "There are " << character_types[0] << " numbers in the input line" << endl;
  cout << "There are " << character_types[1] << " capital letters in the input line" << endl;
  cout << "There are " << character_types[2] << " lower case letters in the input line" << endl;
  cout << "There are " << character_types[3] << " symbols in the input line" << endl;
  cout << "There are " << character_types[4] << " spaces in the input line" << endl;
  cout << "There are " << character_types[5] << " other characters in the input line" << endl;
}

std::vector<std::string> cut_string(std::string x, std::string y) {
  if(!x.length()) {
    throw "The input string should be non-empty!";
  }
  std::vector<string> r;
  std::string s = "";
  r.push_back(s);
  int l = x.length();
  for(int i=0;i<l;i++) {
    if(x.compare(i,1,y))
      r[r.size()-1] += x[i];
    else
      r.push_back(s);
  }
  return r;
}

int main(){
  string mystring;
  getline(cin,mystring);
  //inspect_input(mystring);

  std::string delim = ",";
  try {
    std::vector<string> v = cut_string(mystring,delim);
    cout << v.size() << endl;
    for(int i=0; i<v.size(); i++)
      cout << v[i] << endl;
  }
  catch (const char* msg) {
    cerr << msg << endl;
  }
  
  return 1;
}
