if(NOT "/home/yuqingz/autonomous_driving/baseline/cmake-3.20.0/Tests/CMakeTests" MATCHES "^/")
  set(slash /)
endif()
set(url "file://${slash}/home/yuqingz/autonomous_driving/baseline/cmake-3.20.0/Tests/CMakeTests/FileDownloadInput.png")
set(dir "/home/yuqingz/autonomous_driving/baseline/cmake-3.20.0/Tests/CMakeTests/downloads")

file(DOWNLOAD
  ${url}
  ${dir}/file3.png
  TIMEOUT 2
  STATUS status
  EXPECTED_HASH SHA1=5555555555555555555555555555555555555555
  )
