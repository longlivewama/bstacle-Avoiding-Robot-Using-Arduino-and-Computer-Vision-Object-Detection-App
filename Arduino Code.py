void setup() {
  pinMode(4,OUTPUT);   //Trigger
  pinMode(2,INPUT);    //Echo
  pinMode(5,OUTPUT);   //EnA
  pinMode(6,OUTPUT);   //EnB
  pinMode(8,OUTPUT);   //left motors forward
  pinMode(9,OUTPUT);   //left motors backward
  pinMode(10,OUTPUT);  //right motors forward
  pinMode(11,OUTPUT);  //right motors backward
  Serial.begin(115200);
}

void loop() {
  digitalWrite(4, LOW);
  delayMicroseconds(2);
  digitalWrite(4, HIGH);
  delayMicroseconds(10);
  digitalWrite(4, LOW);
  int duration = pulseIn(2, HIGH);
  int distance = duration* 0.034 /2;
  Serial.println(distance);
  delay(10);

  if(distance >= 20){
    analogWrite(5, 255);
    digitalWrite(8, 1);
    digitalWrite(9, 0);
    analogWrite(6, 255);
    digitalWrite(10, 0);
    digitalWrite(11, 1);
  }
  else if(distance < 20){
    analogWrite(5, 255);
    digitalWrite(8, 1);
    digitalWrite(9, 0);
    analogWrite(6, 255);
    digitalWrite(10, 1);
    digitalWrite(11, 0);
  }

}