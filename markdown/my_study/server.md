## 서버 구축해본 경험 정리
### Linux를 다운로드 받는다.
<li> USB에 linux를 다운받으면 그 usb에는 그 linux만 남고, 다 지워짐. </li>
<li> 서버에 꽂고 reboot하면 알아서 linux를 다운로드 받는다. </li>
<li> 간단한 설정들 모두 다 넘겨주고, 컴퓨터가 켜진다. </li>

### NVIDIA Driver 다운로드.
<li> ubuntu-drivers devices 명령어를 통해 NVIDIA 추천 Driver를 본다. recommended가 쓰여있음. </li>
<img src="https://github.com/wjdwocks/ML-DNN/raw/main/markdown/25년/25.1.22/ubuntu-drivers devices.png" alt="ubuntu-drivers devices" width="500">
<li> 거기서 추천받은 NVIDIA Driver를 다운로드받는다. sudo apt install NVIDIA... 이런식으로 하면 됨. </li>
<li> NVIDIA-Driver와 호환이 되는 CUDA를 다운로드 받아야함. 나는 NVIDIA-Driver 560 이랑 CUDA 12.6으로 다운로드 받음. </li>
<li> (여기서 한번 Driver 다시 다운받으려고 지웠다가 화면 해상도 이상해지고 와이파이도 갑자기 안됨. - wifi문제는 아직도 모르겠다.) </li>
<li> 이렇게 CUDA와 Driver를 모두 다운받은 뒤 Cudnn을 다운받는다. </li>
<li> 인터넷에 cudnn 다운로드 치니까 내꺼 cuda 버전과, 운영체제, 환경 등을 물어보고 최종적으로 나온 bash script만 쭉 입력하니 cudnn이 다운로도 되었다. cuda와 호환 가능한 버전으로. </li>

### anaconda 가상환경 모두 나누기.
<li> 일단 서버에서 anaconda를 다운로드 받았다. </li>
<li> anaconda를 다운받아놓으면 그 명령어를 가지고 가상환경(각 user들마다 각기다른)을 만들어서 각자 관리할 수 있기 때문. </li>
<li> 서버에서 anaconda를 받았는데, 위치가 /home/root/anaconda3 이래가지고, 계정을 만들어도 conda 명령어를 사용할 수가 없었음. </li>
<li> 그래서 우리 연구실 얘들에게 group을 만들어서 거기에 넣은다음, 그 group에게 /home/root/anaconda3의 실행가능한 권한을 주었다. </li>
<li> 그런 후 user 로 로그인 conda conda create --name my_env python=3.9 이런식으로 환경을 각자 만들었다. </li>
<li> 그 다음. conda activate my_env를 하면 이제부터 내 환경으로 들어가서 사용이 가능하다. </li>

---
### 25.1.16 문제 발생
<li> 공인 IP를 받았는데, 그 IP를 설정하고(IP, SubNetMask, Broadcast 등) 인터넷(학교 바닥의 랜선)을 연결했는데, 갑자기 안터넷이 안됨. </li>
<li> 학교 정보전산원에 연락을 해보니, 옆방과 포트가 물려있어서 이쪽이 인터넷이 되면 저쪽이 안되고 하는 상황이었다고 한다. </li>
<li> 그래서 여기서는 정보전산원 분들이 해결을 해줬다.(인터넷 연결 해결) </li>
<li> 그런데, 학교 내에서는 ssh 117.17.199.28 에 접속을 하면 자동으로 22번포트로 잘 연결이 되었는데, 집에서는 같은 방식으로 해도 연결이 안됨. </li>
<li> 생각해보니 유닉스 시스템 수업에서는 ssh -p 22607이런식으로 포트번호를 바꿔서 들어갔던게 생각이 났다. (포트포워딩) </li>
<li> 그래서 검색해보니, 22번포트는 보안 문제 때문에 ISP에서 막아두는 경향이 많다고 해서 3300번 포트를 이용해서 포트포워딩을 진행했다. </li>
<li> 아레와 같이 진행했더니 문제 해결되었다. </li>

### 포트포워딩
<li> 학교에서는 22번 포트로 자동으로 연결이 되었었는데, 집에서 연결하려니까 22번 포트로 연결이 되지 않았음. </li>
<li> 여러가지 검색을 해본 결과 포트포워딩(port forwarding)이 필요한 것 처럼 보였다. </li>
<li> 집에서 telnet 117.17.199.28 22를 해봤을때 연결이 되지 않음. </li>
<li> 검색해보니 학교 ip라서 보안상의 이유로 22번을 막아놓았다는 것 같다. </li>
<li> sudo vi /etc/ssh/sshd_config 를 통해 ssh설정을 보았더니, Port 22와 Port 3300이 둘 다 존재하지만, 22번만 ListenAddress 0.0.0.0으로 되어있고, Port 3300은 #ListenAddress 0.0.0.0으로 주석처리된 것을 확인함. </li>
<li> 저기서 22번을 지우고, 3300의 Listen Address 주석을 지워줌. </li>
<li> sudo systemctl restart ssh를 통해 ssh 서비스 재시작함. </li>
<li> 그 다음 집에서 ssh -p 3300 jaechan8@117.17.199.28을 하니 접속이 됨. </li>

---
### 25.2.10 문제 발생
<li> 친구가 학습을 하던 중에 용량이 부족하다는 에러메시지와 함께 모두의 학습이 중단됨. </li>
<li> du -sh /home/jaechan8/* 와 같이 각자 자기가 사용하는 용량을 봤더니 친구가 batch 단위로 데이터를 저장해서 그런것 같다고 함...... </li>
<li> 암튼, 그 데이터 다 지우고, 정상화 했다. </li>
<li> 근데 처음에 서버를 받고, linux를 다운로드를 했을 때 두 개의 디스크가 있던 것이 기억이 났다. (1TB, 4TB) </li>
<li> 그 때 1TB짜리가 속도가 빠르다는 얘기를 듣고, 그것을 선택했었는데, 4TB가 어떻게 되었는지 서버에서 파일시스템에 들어가 봄. </li>
<li> 근데, 1TB짜리만 인식이 되어있었다. </li>
<li> 그래서 혹시 device가 mount되지 않은건지 확인을 해봤다. </li>


### 디스크 마운팅.
<li> lsblk 를 통해 디스크가 존재하는지와 어디로 마운트되어있는지 확인함. </li>
<li> sda에 3.6T가 있는것을 확인했지만, 파티션도 안나누어져있고, 마운트도 안되어있는것을 확인함. </li>
<li> sudo parted /dev/sda 를 통해 GNU Parted(?)를 열고, mklabel gpt -> mkpart primary ext4 0% 100% -> quit 을 통해 gpt파티션을 이용해 파티션을 만들어줌. </li>
<li> sda1이라는 파티션이 생성됨. </li>
<li> sudo mkfs.ext4 /dev/sda1 을 통해 파일 시스템을 포맷함. </li>
<li> /home/storage 라는 폴더를 생성한 뒤 sda1을 이 디렉토리에 마운트해주었다. </li>
<li> sudo mount /dev/sda1 /home/storage </li>
<li> 리부팅 후에도 자동으로 마운트 되도록 하려면 sudo blkid /dev/sda1 을 통해 uuid를 출력하고, 이를 /etc/fstab에 추가해야한다고 해서 해주었다. </li>

### 사용을 어떻게 하나?
<li> 이 디렉토리를 만들었지만, rwx에 대한 권한이 애메했다. r, x만 있고, w가 없어서 데이터를 넣고, 사용이 안됨. (유저들은 접근도 안됨.) </li>
<li> 그래서, root_access라는 group을 만들어서, 우리 유저들을 넣은 다음, 저 디렉토리에 대한 rwx 권한을 부여해주었다. </li>
<li> 보니, data에 대한 용량이 거의 대부분이고, .py나 .sh파일들, 간단한 .git파일들의 용량은 별로 안크고, 데이터가 대부분이라서, 이 데이터를 저장할 storage 디렉토리를 4TB짜리로 했다. </li>
<li> 이거를 바꾸니, 모든 .py파일의 데이터 경로를 바꾸고 싶지 않아서, symbolic link를 통해 참조만 남겨두는 방법을 사용함. </li>

### 25.4.11 유저 더 추가.
<li> 사람이 둘 더 들어옴. </li>
<li> 서버에 adduser로 추가를 하고, (자동으로)홈디렉토리까지 만들어줌. </li>
<li> Conda를 어차피 서버에 공유로 사용하려고 다운받아놨으니, 환경까지 만들어줘야함. </li>
<li> 아레 명령어만 차례대로 하면 됨. <li>
<img src="https://github.com/wjdwocks/ML-DNN/raw/main/markdown/25년/Study/server.png" alt="user add" width="500">


### 느낀점
<li> 위에 과정을 간추려보니 매우 아무것도 아닌 것 같지만, 아무것도 모른 채로 하려고 하니 죽을맛이었다. </li>
<li> 계속 뭐 하나 하려고 하면 한쪽에서 문제가 생기고 하는게 정말 읍읍마려웠다. </li>
<li> 가만히 냅뒀는데 혼자 갑자기 꺼졌다. 내쪽에서는 server suspend? 이렇게 뜸. - 절전모드 설정되어있나 해서 꺼봄. </li>
