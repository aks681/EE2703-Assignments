#include<stdio.h>
#include<math.h>
int main() {
  double a = 0.2,b;
  double result[1000];
  result[0] = 0.2;
  for (int i = 1;i<1000;i++) {
    b = (a + M_PI)*100 - (int)((a + M_PI)*100);
    a = b;
    result[i] = a;
  }

  for (int i=0;i<1000;i++) {
    printf("%d: %.4f\n",i+1,result[i]);
  }
  return 0;
}
