# 새로운 Django 프로젝트 시작하기 및 Django 기본 구조 + Git

## 목차
- [소개](#소개)
- [준비사항](#준비사항)
- [Django 설치하기](#django-설치하기)
- [Django 프로젝트 생성](#django-프로젝트-생성)
- [Django 기본 구조](#django-기본-구조)
- [Git을 사용한 버전 관리](#git을-사용한-버전-관리)
  - [Git 초기 설정](#git-초기-설정)
  - [Git 기본 개념](#git-기본-개념)
  - [Git 워크플로우](#git-워크플로우)
  - [브랜치 전략](#브랜치-전략)
  - [자주 사용하는 Git 명령어](#자주-사용하는-git-명령어)
  - [Git과 Django 프로젝트](#git과-django-프로젝트)
  - [.gitignore 생성하기](#gitignore-생성하기)
  - [Git 문제 해결하기](#git-문제-해결하기)
- [Cloudflared 터널 사용하기](#cloudflared-터널-사용하기)
- [실습 가이드](#실습-가이드)
- [추가 자료](#추가-자료)

## 소개
이 튜토리얼은 Django 웹 프레임워크의 기본 구조를 이해하고, 새로운 프로젝트를 시작하는 방법을 배우며, Git을 활용하여 효율적으로 프로젝트를 관리하는 방법을 다룹니다. 웹 개발 입문자부터 중급자까지 모두에게 유용한 내용을 담고 있습니다.

## 준비사항
* Python 3.8 이상 설치
* 기본적인 터미널/명령 프롬프트 사용 지식
* 코드 에디터 (VSCode, PyCharm 등)
* Git 설치 (https://git-scm.com/downloads)
* Cloudflared 설치 (https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation)

## Django 설치하기
가상환경을 생성하고 Django를 설치하는 방법:

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows)
venv\Scripts\activate

# 가상환경 활성화 (macOS/Linux)
source venv/bin/activate

# Django 설치
pip install django

# Django 버전 확인
python -m django --version
```

## Django 프로젝트 생성
새 Django 프로젝트를 시작하는 방법:

```bash
# 프로젝트 생성
django-admin startproject myproject

# 프로젝트 디렉토리로 이동
cd myproject

# 애플리케이션 생성
python manage.py startapp myapp

# 서버 실행
python manage.py runserver
```

## Django 기본 구조
Django 프로젝트의 주요 구성 요소:

### 프로젝트 디렉토리 구조
```
myproject/
├── manage.py
├── myproject/
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── myapp/
    ├── __init__.py
    ├── admin.py
    ├── apps.py
    ├── migrations/
    ├── models.py
    ├── tests.py
    └── views.py
```

### 주요 파일 설명
* **settings.py**: 프로젝트 설정 (데이터베이스, 앱 설정, 미들웨어 등)
* **urls.py**: URL 라우팅 설정
* **models.py**: 데이터베이스 모델 정의
* **views.py**: 요청 처리 및 응답 생성 로직
* **templates/**: HTML 템플릿 파일 위치
* **static/**: CSS, JavaScript, 이미지 등 정적 파일 위치

### MVT 패턴
Django는 **MVT(Model-View-Template)** 패턴을 따릅니다:
* **Model**: 데이터 구조 정의 및 데이터베이스 관리
* **View**: 비즈니스 로직 처리
* **Template**: 사용자 인터페이스

## Git을 사용한 버전 관리

### Git 초기 설정
```bash
# Git 사용자 정보 설정 (최초 1회)
git config --global user.name "홍길동"
git config --global user.email "gildong@example.com"

# Git 저장소 초기화
git init

# .gitignore 파일 생성
touch .gitignore
```

### Git 기본 개념
* **저장소(Repository)**: 프로젝트의 모든 파일과 버전 기록을 저장하는 공간
* **커밋(Commit)**: 변경사항의 스냅샷, 프로젝트의 특정 시점 상태를 저장
* **브랜치(Branch)**: 독립적인 작업 라인, 메인 코드에 영향을 주지 않고 기능 개발 가능
* **원격 저장소(Remote)**: 인터넷이나 네트워크 상의 저장소 (GitHub, GitLab 등)
* **작업 디렉토리(Working Directory)**: 실제 작업하는 파일들이 있는 디렉토리
* **스테이징 영역(Staging Area)**: 커밋할 변경사항을 준비하는 영역

### Git 워크플로우
일반적인 Git 작업 흐름:

1. **변경사항 만들기**: 프로젝트 파일 수정
2. **변경사항 스테이징**: `git add` 명령으로 다음 커밋에 포함할 변경사항 선택
3. **변경사항 커밋**: `git commit` 명령으로 스테이징된 변경사항 저장
4. **원격 저장소와 동기화**: `git push`, `git pull` 명령으로 원격 저장소와 동기화

```bash
# 1. 파일 수정 후
# 2. 변경사항 스테이징
git add 파일명    # 특정 파일만 스테이징
git add .        # 모든 변경사항 스테이징

# 3. 변경사항 커밋
git commit -m "기능 구현: 로그인 페이지 추가"

# 4. 원격 저장소에 푸시
git push origin main
```

### 브랜치 전략
효율적인 협업을 위한 브랜치 전략:

#### Git Flow
* **main/master**: 배포 가능한 상태의 코드
* **develop**: 개발 중인 코드의 통합 브랜치
* **feature/**: 새로운 기능 개발 브랜치
* **hotfix/**: 긴급 버그 수정 브랜치
* **release/**: 릴리스 준비 브랜치

```bash
# 새 기능 개발을 위한 브랜치 생성
git checkout -b feature/login-page

# 기능 개발 완료 후 develop 브랜치로 병합
git checkout develop
git merge feature/login-page

# 릴리스 준비
git checkout -b release/v1.0
# 테스트 및 버그 수정 후
git checkout main
git merge release/v1.0
git tag v1.0
```

#### GitHub Flow (간소화된 워크플로우)
* **main**: 항상 배포 가능한 상태 유지
* **feature/**: 모든 개발은 feature 브랜치에서 진행 후 PR(Pull Request)로 병합

```bash
# 새 기능 브랜치 생성
git checkout -b feature/user-authentication

# 개발 완료 후 변경사항 푸시
git push origin feature/user-authentication

# GitHub에서 PR 생성 후 코드 리뷰 및 병합
```

### 자주 사용하는 Git 명령어

#### 기본 명령어
```bash
# 저장소 복제
git clone https://github.com/username/repository.git

# 현재 상태 확인
git status

# 변경 내역 확인
git diff
git diff --staged  # 스테이징된 변경사항만 보기

# 커밋 히스토리 확인
git log
git log --oneline  # 한 줄로 간략하게 보기
git log --graph    # 그래프 형태로 보기

# 특정 파일의 변경 이력 확인
git blame 파일명
```

#### 브랜치 관련 명령어
```bash
# 브랜치 목록 확인
git branch
git branch -a  # 원격 브랜치 포함 모든 브랜치 보기

# 브랜치 생성
git branch 브랜치명

# 브랜치 전환
git checkout 브랜치명
git switch 브랜치명  # Git 2.23 이상

# 브랜치 생성 및 전환
git checkout -b 브랜치명
git switch -c 브랜치명  # Git 2.23 이상

# 브랜치 삭제
git branch -d 브랜치명  # 병합된 브랜치 삭제
git branch -D 브랜치명  # 강제 삭제
```

#### 변경사항 관리
```bash
# 모든 변경사항 취소 (스테이징되지 않은 변경사항)
git restore .
git checkout -- .  # 구버전

# 스테이징된 변경사항 취소
git restore --staged 파일명
git reset HEAD 파일명  # 구버전

# 마지막 커밋 수정
git commit --amend -m "새 커밋 메시지"

# 특정 커밋으로 되돌리기 (새 커밋 생성)
git revert 커밋해시

# 특정 커밋으로 리셋 (히스토리 변경, 주의 필요)
git reset --soft 커밋해시  # 변경사항은 스테이징 상태로 유지
git reset --mixed 커밋해시  # 변경사항은 유지되나 스테이징 취소
git reset --hard 커밋해시  # 변경사항 모두 삭제
```

#### 원격 저장소 관련
```bash
# 원격 저장소 확인
git remote -v

# 원격 저장소 추가
git remote add origin https://github.com/username/repository.git

# 원격 저장소에서 변경사항 가져오기 (병합하지 않음)
git fetch origin

# 원격 저장소에서 변경사항 가져와 병합하기
git pull origin 브랜치명

# 원격 저장소로 변경사항 보내기
git push origin 브랜치명

# 원격 브랜치 삭제
git push origin --delete 브랜치명
```

### Git과 Django 프로젝트

### .gitignore 생성하기

#### gitignore.io 사용하기
[gitignore.io](https://www.toptal.com/developers/gitignore)는 프로젝트 유형에 맞는 `.gitignore` 파일을 쉽게 생성할 수 있는 웹 서비스입니다.

**웹 인터페이스 사용**:
1. [gitignore.io](https://www.toptal.com/developers/gitignore) 웹사이트 방문
2. 검색창에 `django`, `python`, `venv`, `pycharm`(또는 사용 중인 IDE) 등 관련 키워드 입력
3. "Create" 버튼 클릭
4. 생성된 내용을 복사하여 프로젝트의 `.gitignore` 파일에 붙여넣기

**명령줄 사용 (macOS/Linux)**:
```bash
# curl을 사용하여 gitignore 파일 생성
curl -sL https://www.toptal.com/developers/gitignore/api/django,python,venv,visualstudiocode > .gitignore
```

**Windows PowerShell**:
```powershell
# Invoke-WebRequest를 사용하여 gitignore 파일 생성
Invoke-WebRequest -Uri "https://www.toptal.com/developers/gitignore/api/django,python,venv,visualstudiocode" -OutFile ".gitignore"
```

**Git 별칭 설정 (선택사항)**:
자주 사용한다면 Git 별칭을 설정하여 더 쉽게 사용할 수 있습니다:

```bash
# Git 별칭 설정
git config --global alias.ignore '!gi() { curl -sL https://www.toptal.com/developers/gitignore/api/$@ > .gitignore; }; gi'

# 사용 방법
git ignore django,python,venv,visualstudiocode
```

#### Django 프로젝트를 위한 .gitignore
Django 프로젝트에 적합한 .gitignore 파일 기본 내용:

```
# 가상환경
venv/
env/
.env

# Python 캐시 파일
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Django 관련
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal
media/
staticfiles/

# IDE 관련
.idea/
.vscode/
*.swp
*.swo
.DS_Store

# 테스트, 캐시, 커버리지 관련
htmlcov/
.tox/
.coverage
.coverage.*
.cache/
nosetests.xml
coverage.xml
*.cover

# 배포 관련
*.mo
*.pot
```

#### Django 설정 파일 관리
민감한 정보(API 키, 비밀번호 등)를 관리하는 방법:

```python
# settings.py
import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 환경 변수 로드

SECRET_KEY = os.getenv('SECRET_KEY', 'default-secret-key-for-dev')
DEBUG = os.getenv('DEBUG', 'True') == 'True'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('DB_NAME', 'dbname'),
        'USER': os.getenv('DB_USER', 'dbuser'),
        'PASSWORD': os.getenv('DB_PASSWORD', 'dbpassword'),
        'HOST': os.getenv('DB_HOST', 'localhost'),
        'PORT': os.getenv('DB_PORT', '5432'),
    }
}
```

#### 디버그 모드에 따른 설정 분리
```python
# settings/base.py - 공통 설정
# settings/development.py - 개발 환경 설정
# settings/production.py - 운영 환경 설정

# manage.py에서 환경에 맞게 설정 선택
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings.development')
```

### Git 문제 해결하기

#### 충돌 해결하기
```bash
# 충돌이 발생한 경우
git merge feature/branch
# CONFLICT 메시지가 출력됨

# 1. 충돌 파일을 열어서 수동으로 충돌 해결
# <<<<<< HEAD, ======, >>>>>> feature/branch 표시가 있는 부분을 수정

# 2. 충돌 해결 후 파일 스테이징
git add 충돌파일명

# 3. 병합 커밋 생성
git commit
```

#### 커밋 메시지 규칙
의미 있는 커밋 메시지를 작성하는 예시:
```
feat: 로그인 기능 구현
fix: 사용자 프로필 조회 버그 수정
docs: README 업데이트
style: 코드 포맷팅 적용
refactor: 사용자 인증 로직 개선
test: 회원가입 테스트 추가
chore: 패키지 업데이트
```

#### 유용한 Git 팁
* **git stash**: 작업 중인 변경사항을 임시 저장
* **git cherry-pick**: 특정 커밋만 현재 브랜치에 적용
* **git rebase**: 커밋 히스토리를 깔끔하게 정리
* **git bisect**: 버그가 처음 발생한 커밋 찾기

## Cloudflared 터널 사용하기

### Cloudflared란?
Cloudflare Argo Tunnel(현재 Cloudflare Tunnel로 알려짐)은 로컬 서버를 인터넷에 안전하게 노출시킬 수 있는 서비스입니다. `cloudflared`는 이 터널을 설정하는 데 사용되는 명령줄 도구입니다.

### Cloudflared 설치하기

**Windows**:
* [Cloudflared 다운로드 페이지](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation)에서 Windows용 설치 프로그램 다운로드 및 실행

**macOS**:
```bash
brew install cloudflare/cloudflare/cloudflared
```

**Linux (Debian/Ubuntu)**:
```bash
curl -L https://pkg.cloudflare.com/cloudflared-stable-linux-amd64.deb -o cloudflared.deb
sudo dpkg -i cloudflared.deb
```

### Django 서버 터널링하기
Django 서버가 실행 중인 상태에서(일반적으로 `localhost:8000`), 다음 명령어를 실행하여 로컬 서버를 인터넷에 안전하게 노출시킬 수 있습니다:

```bash
cloudflared tunnel --url localhost:8000
```

**주요 특징**:
* 이 명령은 임시 공개 URL을 생성하여 로컬 Django 서버에 접근할 수 있게 해줍니다
* 방화벽 설정이나 포트 포워딩 없이 로컬 개발 서버를 다른 사람과 공유할 수 있습니다
* SSL 암호화를 자동으로 제공합니다

**사용 시나리오**:
* 모바일 기기에서 로컬 개발 서버 테스트
* 클라이언트나 팀원에게 진행 중인 작업 데모
* 웹훅 테스트

### 주의사항
* 이 방법은 개발 및 테스트 목적으로만 사용해야 합니다
* 프로덕션 환경에서는 적절한 배포 방법을 사용하세요
* 터미널을 닫거나 `Ctrl+C`를 누르면 터널이 종료됩니다

## 실습 가이드

### 1. 간단한 블로그 애플리케이션 만들기
1. 모델 정의하기 (models.py)
2. 관리자 페이지 설정 (admin.py)
3. URL 패턴 설정 (urls.py)
4. 뷰 작성하기 (views.py)
5. 템플릿 생성하기 (templates/)
6. 정적 파일 추가하기 (static/)

### 2. 데이터베이스 마이그레이션
```bash
# 모델 변경사항 감지 및 마이그레이션 파일 생성
python manage.py makemigrations

# 마이그레이션 적용
python manage.py migrate
```

### 3. 관리자 계정 생성
```bash
python manage.py createsuperuser
```

### 4. Git으로 프로젝트 버전 관리하기
```bash
# 초기 설정
git init
# gitignore.io 활용하여 .gitignore 생성
curl -sL https://www.toptal.com/developers/gitignore/api/django,python,venv > .gitignore

# 초기 커밋
git add .
git commit -m "초기 프로젝트 설정"

# GitHub 저장소 생성 후
git remote add origin https://github.com/username/myproject.git
git push -u origin main
```

### 5. Cloudflared로 프로젝트 공유하기
```bash
# Django 서버 실행
python manage.py runserver

# 새 터미널에서 cloudflared 실행
cloudflared tunnel --url localhost:8000
```

## 추가 자료

### Git 관련 자료
* [Git 공식 문서](https://git-scm.com/doc)
* [Pro Git 책 (한국어)](https://git-scm.com/book/ko/v2)
* [Git 브랜치 전략](https://nvie.com/posts/a-successful-git-branching-model/)
* [Git 커밋 메시지 작성 가이드](https://chris.beams.io/posts/git-commit/)

### Django 관련 자료
* [Django 공식 문서](https://docs.djangoproject.com/ko/)
* [점프 투 장고](https://wikidocs.net/book/4223)
* [Django REST Framework 공식 문서](https://www.django-rest-framework.org/)

### 커뮤니티 및 포럼
* [Stack Overflow - Django 태그](https://stackoverflow.com/questions/tagged/django)
* [Stack Overflow - Git 태그](https://stackoverflow.com/questions/tagged/git)
* [Django 포럼](https://forum.djangoproject.com/)
* [Reddit r/django](https://www.reddit.com/r/django/)

### 온라인 강의 및 코스
* [Django for Everybody](https://www.dj4e.com/) - 미시간 대학교의 Django 강의
* [MDN Django 튜토리얼](https://developer.mozilla.org/ko/docs/Learn/Server-side/Django)
* [Codecademy Git 코스](https://www.codecademy.com/learn/learn-git)
* [Udemy Django 강의](https://www.udemy.com/topic/django/)
